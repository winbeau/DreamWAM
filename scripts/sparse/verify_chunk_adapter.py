#!/usr/bin/env python3
"""Verify actual Dense/M1-M3 graph adapters against causal uncached eager replays."""

import argparse
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from benchmark_ffn_context_cache import command
from observation_sequences import load_sequences, sha256
from evaluation.action_eval.infer import create_policy
from dreamwam.sparse.conditioned_frame_cache import ConditionedFrameCache
from dreamwam.sparse.hybrid.runtime import HybridVisualRuntime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--options", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("select one authorized CUDA device")
    if command("git", "status", "--porcelain")["stdout"]:
        raise RuntimeError("adapter verification requires committed clean source")
    manifest, sequences = load_sequences(args.inputs)
    candidate = json.loads(args.options.read_text())
    root = Path(__file__).resolve().parents[2]
    common = dict(action_horizon=32, denoising_steps=10, rng_mode="fixed_per_predict", prompt_cache={"capacity": 8})
    variants = dict(dense=dict(**common, visual_cache=dict(refresh_every=1,
        conditioned_frame_reuse=True, graph_dispatch="all_transformers")), sparse={**common, **candidate})
    args.out_dir.mkdir(parents=True, exist_ok=False)
    report = dict(status="RUNNING", start_utc=datetime.now(timezone.utc).isoformat(), argv=sys.argv,
        commit=command("git", "rev-parse", "HEAD"), options_sha256=sha256(args.options),
        inputs_sha256=sha256(args.inputs), input_split_sha256=manifest["episode_split_sha256"],
        torch=torch.__version__, cuda=torch.version.cuda, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        gpu_name=torch.cuda.get_device_name(), arms={}, sr=None,
        limitations=["real-checkpoint adapter equivalence, not closed-loop success or a latency benchmark"])
    write = lambda: (args.out_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    adapter = None
    try:
        write()
        for arm, options in variants.items():
            adapter = create_policy(dict(model_root=str(root), model_config=str(root / "configs/dreamwam_joint.yaml"),
                checkpoint=str(root / "checkpoints/dreamwam_joint.pt"), **options), "cuda")
            fingerprint = adapter.describe().fingerprint
            if fingerprint["checkpoint_sha256"] != "6c087e5b9e201f19dbe92834b470a7ee18937a18ea9adcbb49886aa1e3ac4c61":
                raise AssertionError("checkpoint differs from the frozen reference")
            policy = adapter.policy
            versions = {name: value._version for name, value in policy.model.named_parameters()}
            evaluation = dict(policy.evaluation)
            attr = "_hybrid_visual_runtime" if arm == "sparse" else "_visual_cache_runtime"
            diagnostics_key = "hybrid_visual" if arm == "sparse" else "visual_cache"
            runtime = getattr(policy, attr)
            prompt = policy._prompt_cache_runtime
            runtime.__exit__(None, None, None)
            eager = (HybridVisualRuntime(policy.model, replace(runtime.base_config, backend="eager"))
                     if arm == "sparse" else ConditionedFrameCache(policy.model, refresh_every=1))
            refs, saved = {}, {}
            setattr(policy, attr, eager)
            policy.text_encoder = prompt.encoder
            try:
                with eager:
                    for identity, sequence in sequences.items():
                        adapter.reset(dict(episode_id=str(identity), seed=identity[-1]))
                        for entry, observation in sequence:
                            prediction = adapter.predict(observation)
                            refs[entry["id"]] = (prediction.actions.copy(), deepcopy(prediction.diagnostics))
                            saved["reference_" + entry["id"]] = prediction.actions.copy()
            finally:
                setattr(policy, attr, runtime)
                policy.text_encoder = prompt
                runtime.__enter__()
            record = report["arms"][arm] = dict(fingerprint=fingerprint, options=options, requests=[])
            for repeat in range(2):
                for identity, sequence in sequences.items():
                    adapter.reset(dict(episode_id=str(identity), seed=identity[-1]))
                    if prompt.entries:
                        raise AssertionError("prompt cache survived episode reset")
                    for index, (entry, observation) in enumerate(sequence):
                        torch.cuda.synchronize(); started = time.perf_counter()
                        prediction = adapter.predict(observation)
                        torch.cuda.synchronize(); elapsed = time.perf_counter() - started
                        reference, reference_stats = refs[entry["id"]]
                        if not np.array_equal(prediction.actions, reference):
                            raise AssertionError("graph adapter differs from causal eager reference: " + entry["id"])
                        actual = prediction.diagnostics
                        expected = reference_stats[diagnostics_key]
                        work = actual[diagnostics_key]
                        if work["action_layer_updates"] != 300 or work["computed_video_token_layers"] != expected["computed_video_token_layers"]:
                            raise AssertionError("action updates or executed visual work changed")
                        if actual["prompt_cache"]["last_hit"] != (index > 0):
                            raise AssertionError("unexpected instruction-cache boundary")
                        if arm == "sparse":
                            if work["read_video_token_layers"] != expected["read_video_token_layers"]:
                                raise AssertionError("read budget differs from eager")
                            if [s["effective_op"] for s in work["steps"]] != [s["effective_op"] for s in expected["steps"]]:
                                raise AssertionError("adaptive operations differ from eager")
                            if runtime.state.output is not None or runtime._active:
                                raise AssertionError("visual state leaked across chunks")
                            for key in ("level_index", "budget", "features", "reason", "history_length", "chunk_index"):
                                if actual["chunk_budget"][key] != reference_stats["chunk_budget"][key]:
                                    raise AssertionError("causal M1 decision differs from eager")
                        record["requests"].append(dict(input_id=entry["id"], repeat=repeat,
                            seconds_including_setup=elapsed, bitwise_own_uncached_eager=True, diagnostics=actual))
                        saved[f"actual_{repeat}_" + entry["id"]] = prediction.actions.copy()
                    print(json.dumps(dict(arm=arm, repeat=repeat, episode=identity, calls=len(sequence), bitwise=True)), flush=True)
                    write()
            if fingerprint != adapter.describe().fingerprint:
                raise AssertionError("stable policy identity changed with online budgets")
            if evaluation != policy.evaluation or versions != {n: p._version for n, p in policy.model.named_parameters()}:
                raise AssertionError("model weights or evaluation protocol changed")
            np.savez_compressed(args.out_dir / (arm + "-actions.npz"), **saved)
            record["actions_sha256"] = sha256(args.out_dir / (arm + "-actions.npz"))
            adapter.close(); adapter = None
            record["close_released_caches"] = not prompt.entries and not (
                runtime.dispatch.entries if arm == "sparse" else runtime.graphs)
            if not record["close_released_caches"]:
                raise AssertionError("close retained model graph or prompt caches")
            del policy, runtime, eager, prompt
            gc.collect(); torch.cuda.empty_cache()
            write()
        report.update(status="VERIFIED", exit_code=0)
    except BaseException as exc:
        report.update(status="ERROR", error=repr(exc), exit_code=1)
        raise
    finally:
        if adapter is not None: adapter.close()
        report["end_utc"] = datetime.now(timezone.utc).isoformat()
        write()
    print(json.dumps(dict(status=report["status"], calls=sum(len(a["requests"]) for a in report["arms"].values()), sr=None)))


if __name__ == "__main__":
    main()
