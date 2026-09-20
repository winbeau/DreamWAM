#!/usr/bin/env python3
"""Measure visual refresh interval 5 versus 10 at fixed common optimizations.

Conditioned-frame reuse and transformer graph scope are fixed on both sides.
All ten action denoising steps remain. Native Dense and equally optimized Dense
are measured in the same balanced full-request schedule. Synthetic inputs only;
the output cannot establish closed-loop SR or decision preservation.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

from benchmark_ffn_context_cache import command, inputs_for, sha256, summary
from dreamwam.config import load_release_config
from dreamwam.policy import build_policy
from dreamwam.sparse.conditioned_frame_cache import ConditionedFrameCache, GraphedConditionedFrameCache
from dreamwam.sparse.visual_step_cache import VisualStepCache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dreamwam_joint.yaml")
    parser.add_argument("--reps", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    if args.reps < 21 or args.reps % 21 or args.warmup < 1:
        parser.error("use a positive multiple of 21 repeats to balance seven positions and three inputs")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    release = load_release_config(args.config)
    inventory = lambda: command("nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,memory.used", "--format=csv")
    manifest = dict(status="RUNNING", start_utc=datetime.now(timezone.utc).isoformat(),
        argv=sys.argv, cwd=str(Path.cwd()), pid=os.getpid(),
        git=command("git", "rev-parse", "HEAD"), dirty=command("git", "status", "--porcelain"),
        python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), gpu_before=inventory(),
        checkpoint_sha256=sha256(release.paths.checkpoint),
        sources={name: sha256(name) for name in ("dreamwam/sparse/conditioned_frame_cache.py",
            "dreamwam/sparse/visual_step_cache.py", "dreamwam/sparse/visual_token_cache.py",
            "dreamwam/sparse/visual_cache_graphs.py", "scripts/sparse/benchmark_visual_cadence.py",
            "pyproject.toml", "requirements.txt", args.config)},
        factor="visual refresh interval 5 versus 10; unchanged conditioned-frame reuse and graph scope",
        environment="existing Python/torch environment reused unchanged; no install or sync",
        latency_boundary="synchronized full predict_action through CPU action output", sr=None)
    write = lambda: (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write()
    caches = {}
    try:
        started = time.perf_counter()
        policy = build_policy(release, device="cuda")
        evaluation = dict(policy.evaluation)
        if evaluation["denoising_steps"] != 10 or evaluation["action_horizon"] != 32:
            raise ValueError("this cadence experiment requires the unchanged released ten steps and horizon 32")
        manifest.update(model_load_seconds=time.perf_counter() - started,
                        evaluation=evaluation, dtype=str(policy.dtype),
                        fast_ops_enabled=policy.model.fast_ops_enabled)
        versions = {name: parameter._version for name, parameter in policy.model.named_parameters()}
        intervals = {"dense": 1, "r5": 5, "r10": 10}
        for kind, factory in (("eager", ConditionedFrameCache), ("graph", GraphedConditionedFrameCache)):
            for label, interval in intervals.items():
                caches[f"{kind}_{label}"] = factory(policy.model, refresh_every=interval)
        variants = ("native_dense", *caches)
        inputs, provenance = inputs_for(argparse.Namespace(inputs_npz=None), policy.image_size)
        inputs.append((inputs[0][0], np.full(8, 0.05, dtype=np.float32),
                       "pick up the red mug and place it in the basket"))
        manifest.update(inputs=provenance, extra_parity_input="input 0 images, state=0.05, red mug instruction",
                        reps_per_variant=args.reps, warmup_per_variant=args.warmup, graph_warmup=3,
                        variants=variants, cold_requests=[], parity_requests=[])
        references, preconditioned = {}, {}

        def predict(name, input_id, override=None):
            images, state, instruction = inputs[input_id]
            with (override if override is not None else caches.get(name, nullcontext())) as cache:
                torch.cuda.synchronize()
                started = time.perf_counter()
                actions = policy.predict_action(images=images, state=state, instruction=instruction)
                torch.cuda.synchronize()
                seconds = time.perf_counter() - started
                stats = dict(cache.last_stats) if cache is not None else {}
            if not np.isfinite(actions).all():
                raise FloatingPointError(name)
            return seconds, actions, stats

        def differences(name, input_id, actions):
            dense = references["native_dense", input_id]
            delta = actions.astype(np.float64) - dense.astype(np.float64)
            label = name.split("_")[-1]
            before = dense if label == "dense" else preconditioned[label, input_id]
            return dict(bitwise_dense=bool(np.array_equal(actions, dense)),
                        max_abs_dense=float(np.abs(delta).max()),
                        relative_l2_dense=float(np.linalg.norm(delta) / max(np.linalg.norm(dense), 1e-12)),
                        bitwise_before_conditioned_reuse=bool(np.array_equal(actions, before)))

        def check_counts(name, stats):
            if name == "native_dense":
                return
            label = name.split("_")[-1]
            partials = 9 // intervals[label]
            if (stats["dense_video_steps"] != 1 or stats["partial_video_steps"] != partials or
                stats["reused_video_steps"] != 9 - partials or
                stats["conditioned_token_layer_reuses"] != 98 * 30 * partials or
                stats["action_layer_updates"] != 300):
                raise AssertionError(f"unexpected visual cadence or action steps: {name}")

        for request, input_id in enumerate((0, 1, 2, 3, 0)):
            for label in ("r5", "r10"):
                _, actions, _ = predict("reference", input_id,
                    override=VisualStepCache(policy.model, refresh_every=intervals[label]))
                key = (label, input_id)
                if key in preconditioned and not np.array_equal(actions, preconditioned[key]):
                    raise AssertionError("preconditioned reference changed across requests")
                preconditioned[key] = actions.copy()
            for name in variants:
                seconds, actions, stats = predict(name, input_id)
                own = name.replace("graph_", "eager_")
                key = (own, input_id)
                if own == name and key not in references:
                    references[key] = actions.copy()
                if not np.array_equal(actions, references[key]):
                    raise AssertionError(f"own-eager parity/request isolation failed: {name}, input {input_id}")
                check_counts(name, stats)
                row = dict(variant=name, request=request, input_id=input_id,
                           bitwise_own_eager=True, **differences(name, input_id, actions))
                manifest["parity_requests"].append(row)
                if request == 0:
                    manifest["cold_requests"].append(dict(variant=name, seconds=seconds, cache=stats))
                write()
            print(f"request {request}, input {input_id}: all seven own-eager and computation-count checks passed", flush=True)
        for name, cache in caches.items():
            if name.startswith("graph_"):
                required = {"dense"}
                if name != "graph_dense":
                    required.add("action")
                if name != "graph_r10":
                    required.add("partial")
                graphs = cache.graph_stats()
                if set(graphs) != required or not all(row["captured"] for row in graphs.values()):
                    raise AssertionError(f"missing transformer graph capture: {name}")
        for name in variants:
            for rep in range(args.warmup):
                predict(name, rep % 3)
        samples = {name: [] for name in variants}
        balance = Counter()
        with (out / "requests.jsonl").open("w", buffering=1) as raw:
            for rep in range(args.reps):
                order = variants[rep % 7:] + variants[:rep % 7]
                input_id = rep % 3
                for position, name in enumerate(order):
                    seconds, actions, stats = predict(name, input_id)
                    own = name.replace("graph_", "eager_")
                    if not np.array_equal(actions, references[own, input_id]):
                        raise AssertionError(f"timed own-eager parity failed: {name}")
                    check_counts(name, stats)
                    balance[name, position, input_id] += 1
                    samples[name].append(seconds)
                    raw.write(json.dumps(dict(repeat=rep, position=position, order=order, variant=name,
                        input_id=input_id, seconds=seconds, bitwise_own_eager=True, cache=stats,
                        **differences(name, input_id, actions))) + "\n")
                print(f"repeat {rep}: " + " ".join(f"{name}={samples[name][-1]*1000:.2f}ms" for name in order), flush=True)
        if len(balance) != 7 * 7 * 3 or set(balance.values()) != {args.reps // 21}:
            raise AssertionError("not every variant, position and input was equally measured")
        if versions != {name: parameter._version for name, parameter in policy.model.named_parameters()}:
            raise AssertionError("parameter versions changed")
        if policy.evaluation != evaluation:
            raise AssertionError("evaluation settings changed")
        metrics = {name: summary(values) for name, values in samples.items()}
        means = {name: values["mean"] for name, values in metrics.items()}
        report = dict(status="MEASURED", end_utc=datetime.now(timezone.utc).isoformat(), seconds=metrics,
            cadence_factor_eager=means["eager_r5"] / means["eager_r10"],
            cadence_factor_graph=means["graph_r5"] / means["graph_r10"],
            r5_vs_matched_dense=means["graph_dense"] / means["graph_r5"],
            r10_vs_matched_dense=means["graph_dense"] / means["graph_r10"],
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
            graphs={name: cache.graph_stats() for name, cache in caches.items() if name.startswith("graph_")},
            bitwise_own_eager=True, all_bitwise_before_conditioned_reuse=all(
                row["bitwise_before_conditioned_reuse"] for row in manifest["parity_requests"]),
            samples_per_variant_position_input=args.reps // 21, gpu_after=inventory(), sr=None)
        (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        manifest.update(status="MEASURED", end_utc=report["end_utc"])
        write()
        print(json.dumps(report, indent=2), flush=True)
    except BaseException as exc:
        manifest.update(status="ERROR", error=repr(exc), end_utc=datetime.now(timezone.utc).isoformat())
        write()
        raise
    finally:
        for cache in caches.values():
            if hasattr(cache, "close_graphs"):
                cache.close_graphs()


if __name__ == "__main__":
    main()
