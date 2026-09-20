#!/usr/bin/env python3
"""Observe the conditioned-frame invariant without changing model computation.

This is a diagnostic, not a cache implementation or latency/SR measurement.
Every video layer's first-frame Q/K/V and outputs are compared with step zero.
Future-frame changes serve as a positive control. Full actions must be bitwise
equal to a separate, uninstrumented native Dense call on the same input.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from functools import wraps
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

from benchmark_ffn_context_cache import command, inputs_for, sha256
from dreamwam.config import load_release_config
from dreamwam.policy import build_policy


class FrameAudit:
    def __init__(self, model):
        self.model = model
        self.layers = {id(block): i for i, block in enumerate(model.video_expert.blocks)}
        self.originals = []
        self.anchors = {}
        self.rows = []
        self.steps = []
        self.step = None
        self.prefix = None
        self.length = None

    def compare(self, field, tensor, *, layer=None, full=False):
        if self.step is None:
            raise RuntimeError("observed a tensor outside the transformer step")
        key = (field, layer)
        if self.step == 0:
            if key in self.anchors:
                raise AssertionError(f"duplicate step-zero observation: {key}")
            self.anchors[key] = tensor.detach().clone()
            return
        reference = self.anchors[key]
        if tensor.shape != reference.shape or tensor.dtype != reference.dtype:
            raise AssertionError(f"layout changed: {key}")
        ranges = (("context", tensor, reference),) if full else (
            ("conditioned", tensor[:, :self.prefix], reference[:, :self.prefix]),
            ("future", tensor[:, self.prefix:], reference[:, self.prefix:]),
        )
        for region, actual, expected in ranges:
            if not actual.numel():
                raise AssertionError("empty audit region")
            different = int(torch.count_nonzero(actual != expected).item())
            maximum = float((actual.float() - expected.float()).abs().max().item())
            self.rows.append(dict(step=self.step, layer=layer, field=field, region=region,
                                  elements=actual.numel(), changed_elements=different,
                                  max_abs=maximum, bitwise_equal=different == 0))

    def __enter__(self):
        mot = self.model.mot
        forward, attention, post = mot.forward, mot._attention_input, mot._post_attention
        router = self.model.world_residual
        world_forward = router.forward

        @wraps(forward)
        def observe_forward(*, video_state, action_state, residual_injection,
                            sparse=None, step_index=0, num_steps=1):
            if sparse is not None and sparse.enabled:
                raise ValueError("audit requires native Dense")
            if residual_injection is not router:
                raise ValueError("audit requires the original world router")
            self.step = step_index
            self.steps.append(step_index)
            self.prefix = int(video_state["tokens_per_frame"])
            self.length = int(video_state["tokens"].shape[1])
            if not 0 < self.prefix < self.length:
                raise AssertionError("audit requires conditioned and future frames")
            mask = mot.build_attention_mask(video_length=self.length,
                action_length=action_state["tokens"].shape[1],
                video_tokens_per_frame=self.prefix, device=video_state["tokens"].device)
            if bool(mask[:self.prefix, self.prefix:].any()):
                raise AssertionError("conditioned queries can see changing video or actions")
            for field in ("tokens", "time_modulation", "time_embedding"):
                self.compare("pre_" + field, video_state[field])
            self.compare("video_context", video_state["context"], full=True)
            self.compare("action_context", action_state["context"], full=True)
            result = forward(video_state=video_state, action_state=action_state,
                residual_injection=residual_injection, sparse=sparse,
                step_index=step_index, num_steps=num_steps)
            self.compare("final_video", result["video"])
            return result

        @wraps(attention)
        def observe_attention(block, *args, **kwargs):
            result = attention(block, *args, **kwargs)
            if id(block) in self.layers:
                for field, tensor in zip(("query", "key", "value", "layer_input"), result[:4]):
                    self.compare(field, tensor, layer=self.layers[id(block)])
            return result

        @wraps(post)
        def observe_post(block, *args, **kwargs):
            result = post(block, *args, **kwargs)
            if id(block) in self.layers:
                self.compare("post_attention_ffn", result, layer=self.layers[id(block)])
            return result

        @wraps(world_forward)
        def observe_world(layer_index, *args, **kwargs):
            result = world_forward(layer_index, *args, **kwargs)
            self.compare("post_world", result, layer=layer_index)
            return result

        self.originals = [(mot, "forward", forward), (mot, "_attention_input", attention),
                          (mot, "_post_attention", post), (router, "forward", world_forward)]
        mot.forward, mot._attention_input, mot._post_attention = observe_forward, observe_attention, observe_post
        router.forward = observe_world
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for owner, name, original in reversed(self.originals):
            setattr(owner, name, original)
        self.originals.clear()
        self.anchors.clear()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dreamwam_joint.yaml")
    parser.add_argument("--inputs-npz", action="append")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    release = load_release_config(args.config)
    manifest = dict(status="RUNNING", start_utc=datetime.now(timezone.utc).isoformat(),
        argv=sys.argv, git=command("git", "rev-parse", "HEAD"), dirty=command("git", "status", "--porcelain"),
        python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        checkpoint_sha256=sha256(release.paths.checkpoint),
        sources={name: sha256(name) for name in ("dreamwam/model.py", "dreamwam/experts.py",
            "dreamwam/mot.py", "scripts/sparse/audit_conditioned_frame.py", args.config)},
        gpu_before=command("nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,memory.used", "--format=csv"),
        environment="existing environment unchanged; no install or sync",
        scope="native Dense observation only; no cached computation, no latency or SR claim", sr=None)
    write = lambda: (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write()
    try:
        started = time.perf_counter()
        policy = build_policy(release, device="cuda")
        manifest.update(model_load_seconds=time.perf_counter() - started, evaluation=dict(policy.evaluation),
                        dtype=str(policy.dtype), fast_ops_enabled=policy.model.fast_ops_enabled)
        if release.setting != "joint" or policy.model.config.patch_size[0] != 1:
            raise ValueError("audit requires Joint with temporal patch size one")
        versions = {key: p._version for key, p in policy.model.named_parameters()}
        inputs, source = inputs_for(args, policy.image_size)
        inputs.append((inputs[0][0], np.full(8, 0.05, dtype=np.float32),
                       "pick up the red mug and place it in the basket"))
        manifest["inputs"] = source
        reports = []
        with (out / "comparisons.jsonl").open("w", buffering=1) as raw:
            for request, input_id in enumerate((*range(len(inputs)), 0)):
                images, state, instruction = inputs[input_id]
                reference = policy.predict_action(images=images, state=state, instruction=instruction)
                with FrameAudit(policy.model) as audit:
                    actions = policy.predict_action(images=images, state=state, instruction=instruction)
                if not np.isfinite(actions).all() or not np.array_equal(reference, actions):
                    raise AssertionError("instrumentation changed the native Dense actions")
                if audit.steps != list(range(10)):
                    raise AssertionError(f"expected all ten denoising steps: {audit.steps}")
                counts = Counter((row["field"], row["region"]) for row in audit.rows)
                for field in ("query", "key", "value", "layer_input", "post_attention_ffn", "post_world"):
                    if counts[field, "conditioned"] != 9 * policy.model.mot.num_layers:
                        raise AssertionError(f"incomplete layer audit: {field}")
                invariant = [row for row in audit.rows if row["region"] != "future"]
                future = [row for row in audit.rows if row["region"] == "future"]
                for row in audit.rows:
                    raw.write(json.dumps(dict(request=request, input_id=input_id, **row)) + "\n")
                report = dict(request=request, input_id=input_id, video_tokens=audit.length,
                    conditioned_tokens=audit.prefix, steps=audit.steps, layers=policy.model.mot.num_layers,
                    instrumented_action_bitwise=True, comparisons=len(audit.rows),
                    invariant_comparisons=len(invariant), invariant_differences=sum(not r["bitwise_equal"] for r in invariant),
                    invariant_max_abs=max(r["max_abs"] for r in invariant),
                    future_differences=sum(not r["bitwise_equal"] for r in future))
                if report["future_differences"] == 0:
                    raise AssertionError("future-frame positive control did not change")
                reports.append(report)
                manifest["requests"] = reports
                write()
                print(json.dumps(report), flush=True)
        if versions != {key: p._version for key, p in policy.model.named_parameters()}:
            raise AssertionError("parameter versions changed")
        manifest.update(status="OBSERVED", all_conditioned_bitwise=all(r["invariant_differences"] == 0 for r in reports),
                        end_utc=datetime.now(timezone.utc).isoformat())
        write()
    except BaseException as exc:
        manifest.update(status="ERROR", error=repr(exc), end_utc=datetime.now(timezone.utc).isoformat())
        write()
        raise


if __name__ == "__main__":
    main()
