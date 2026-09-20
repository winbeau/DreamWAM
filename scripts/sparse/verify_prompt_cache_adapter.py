#!/usr/bin/env python3
"""Check the measured Dense/Sparse paths through the real action-eval adapter.

Uses the real checkpoint and captured observations. Every adapter output must be
bitwise its own uncached eager reference, including instruction changes, image
changes under a repeated instruction, and the first call after episode reset.
This is an integration check, not closed-loop SR or another timing benchmark.
"""

import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from benchmark_ffn_context_cache import command, sha256
from evaluation.action_eval.infer import create_policy
from dreamwam.sparse.action_guided_visual_token_cache import ActionGuidedVisualTokenCache
from dreamwam.sparse.conditioned_frame_cache import ConditionedFrameCache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    report = dict(status="RUNNING", start_utc=datetime.now(timezone.utc).isoformat(),
        argv=sys.argv, pid=os.getpid(), commit=command("git", "rev-parse", "HEAD"),
        torch=torch.__version__, cuda=torch.version.cuda,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        input_manifest_sha256=sha256(args.inputs), arms={}, sr=None)
    write = lambda: (args.out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    write()
    try:
        data = json.loads(args.inputs.read_text())
        if data["status"] != "CAPTURED" or len(data["inputs"]) != 3:
            raise ValueError("require the three real captured inputs")
        inputs = []
        for item in data["inputs"]:
            path = args.inputs.parent / item["path"]
            if sha256(path) != item["sha256"]:
                raise ValueError("input bytes changed")
            with np.load(path, allow_pickle=False) as values:
                inputs.append(dict(images={k: values[k].copy() for k in ("agentview", "wrist")},
                    state=values["state"].copy(), instruction=str(values["instruction"].item())))
        inputs.append({**inputs[1], "instruction": inputs[0]["instruction"]})
        configs = dict(
            dense=dict(refresh_every=1, conditioned_frame_reuse=True, graph_dispatch="all_transformers"),
            sparse=dict(refresh_every=5, token_keep_ratio=0.1, action_guidance_weight=1.0,
                        graph_dispatch="all_transformers"))
        for arm, options in configs.items():
            adapter = None
            try:
                adapter = create_policy(dict(model_root=str(root),
                    model_config=str(root / "configs/dreamwam_joint.yaml"),
                    checkpoint=str(root / "checkpoints/dreamwam_joint.pt"),
                    visual_cache=options, prompt_cache={"capacity": 8}), "cuda:0")
                description = adapter.describe()
                if description.fingerprint["visual_cache"] != options or description.fingerprint["prompt_cache"] != {"capacity": 8}:
                    raise AssertionError("effective options differ from requested options")
                policy = adapter.policy
                version = {name: p._version for name, p in policy.model.named_parameters()}
                evaluation = dict(policy.evaluation)
                if evaluation["denoising_steps"] != 10 or evaluation["action_horizon"] != 32:
                    raise AssertionError("released action sampling changed")
                runtime, prompt = policy._visual_cache_runtime, policy._prompt_cache_runtime
                runtime.__exit__(None, None, None)
                policy.text_encoder = prompt.encoder
                eager = (ConditionedFrameCache(policy.model, refresh_every=1) if arm == "dense" else
                         ActionGuidedVisualTokenCache(policy.model, refresh_every=5,
                                                     keep_ratio=0.1, guidance_weight=1.0))
                try:
                    with eager:
                        refs = [policy.predict_action(**item).copy() for item in inputs]
                finally:
                    policy.text_encoder = prompt
                    runtime.__enter__()
                record = report["arms"][arm] = dict(fingerprint=description.fingerprint, requests=[])
                saved = {f"reference_{i}": value for i, value in enumerate(refs)}
                adapter.reset(dict(episode_id="parity-episode-0", seed=42))
                sequence = [(0, False), (0, True), (1, False), (2, False), (3, True), (0, True), (0, False)]
                for index, (input_id, expected_hit) in enumerate(sequence):
                    if index == len(sequence) - 1:
                        adapter.reset(dict(episode_id="parity-episode-1", seed=42))
                        if prompt.entries:
                            raise AssertionError("episode reset kept prompt entries")
                    torch.cuda.synchronize(); started = time.perf_counter()
                    prediction = adapter.predict(inputs[input_id])
                    torch.cuda.synchronize(); seconds = time.perf_counter() - started
                    if not np.array_equal(prediction.actions, refs[input_id]):
                        raise AssertionError("adapter differs from its uncached eager reference")
                    if prediction.diagnostics["prompt_cache"]["last_hit"] != expected_hit:
                        raise AssertionError("instruction cache hit/miss boundary changed")
                    stats = prediction.diagnostics["visual_cache"]
                    if stats["action_layer_updates"] != 300 or stats["computed_video_token_layers"] != (61740 if arm == "dense" else 9720):
                        raise AssertionError("executed budget changed")
                    saved[f"adapter_{index}"] = prediction.actions.copy()
                    record["requests"].append(dict(input_id=input_id, seconds_including_setup=seconds,
                        bitwise_own_uncached_eager=True, diagnostics=prediction.diagnostics))
                    write()
                if evaluation != policy.evaluation or version != {n: p._version for n, p in policy.model.named_parameters()}:
                    raise AssertionError("sampling settings or weights changed")
                np.savez_compressed(args.out_dir / f"{arm}-actions.npz", **saved)
                record["actions_sha256"] = sha256(args.out_dir / f"{arm}-actions.npz")
                record["graphs"] = runtime.graph_stats()
                adapter.close(); adapter = None
                if runtime.graphs or prompt.entries:
                    raise AssertionError("close did not release caches")
                record["close_released_caches"] = True
                del runtime, prompt, eager, policy
                gc.collect(); torch.cuda.empty_cache()
                write()
            finally:
                if adapter is not None:
                    adapter.close()
        report.update(status="VERIFIED", end_utc=datetime.now(timezone.utc).isoformat())
        write(); print(json.dumps({k: v for k, v in report.items() if k != "arms"}, indent=2))
    except BaseException as exc:
        report.update(status="ERROR", error=repr(exc), end_utc=datetime.now(timezone.utc).isoformat())
        write(); raise


if __name__ == "__main__":
    main()
