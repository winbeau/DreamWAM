#!/usr/bin/env python3
"""One factor: reuse conditioned-frame rows at an unchanged visual cadence.

Measure full requests for native Dense and the existing interval-5 temporal
candidate, each with and without conditioned-frame reuse. Graph scope is fixed
in the graph comparison. Every graph must be bitwise its own eager reference;
numerical differences from the pre-factor path are reported, not concealed.
"""

from __future__ import annotations

import argparse
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
from dreamwam.sparse.dense_graph_control import NativeDenseGraph
from dreamwam.sparse.visual_cache_graphs import GraphedVisualTokenCache
from dreamwam.sparse.visual_step_cache import VisualStepCache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dreamwam_joint.yaml")
    parser.add_argument("--reps", type=int, default=24)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    if args.reps < 24 or args.reps % 24 or args.warmup < 1:
        parser.error("use a positive multiple of 24 repetitions to balance all eight positions and three inputs")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    release = load_release_config(args.config)
    inventory = lambda: command("nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,memory.used", "--format=csv")
    manifest = dict(status="RUNNING", start_utc=datetime.now(timezone.utc).isoformat(),
        argv=sys.argv, git=command("git", "rev-parse", "HEAD"), dirty=command("git", "status", "--porcelain"),
        python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), gpu_before=inventory(),
        checkpoint_sha256=sha256(release.paths.checkpoint),
        sources={name: sha256(name) for name in ("dreamwam/sparse/conditioned_frame_cache.py",
            "dreamwam/sparse/visual_token_cache.py", "dreamwam/sparse/visual_cache_graphs.py",
            "dreamwam/sparse/dense_graph_control.py", "scripts/sparse/benchmark_conditioned_frame.py", args.config)},
        environment="existing Python/torch environment reused; no install or sync",
        factor="reuse only invariant conditioned-frame rows at fixed visual refresh cadence and graph scope",
        latency_boundary="synchronized full predict_action through CPU action output", sr=None)
    write = lambda: (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write()
    caches = {}
    try:
        started = time.perf_counter()
        policy = build_policy(release, device="cuda")
        manifest.update(model_load_seconds=time.perf_counter() - started,
                        evaluation=dict(policy.evaluation), dtype=str(policy.dtype),
                        fast_ops_enabled=policy.model.fast_ops_enabled)
        versions = {name: parameter._version for name, parameter in policy.model.named_parameters()}
        caches = {
            "eager_temporal": VisualStepCache(policy.model, refresh_every=5),
            "conditioned_dense": ConditionedFrameCache(policy.model, refresh_every=1),
            "conditioned_temporal": ConditionedFrameCache(policy.model, refresh_every=5),
            "graph_dense": NativeDenseGraph(policy.model),
            "graph_temporal": GraphedVisualTokenCache(policy.model, keep_ratio=1.0,
                refresh_every=5, guidance_weight=1.0, graph_partial=True),
            "graph_conditioned_dense": GraphedConditionedFrameCache(policy.model, refresh_every=1),
            "graph_conditioned_temporal": GraphedConditionedFrameCache(policy.model, refresh_every=5),
        }
        variants = ("native_dense", *caches)
        inputs, provenance = inputs_for(argparse.Namespace(inputs_npz=None), policy.image_size)
        inputs.append((inputs[0][0], np.full(8, 0.05, dtype=np.float32),
                       "pick up the red mug and place it in the basket"))
        manifest.update(inputs=provenance, extra_parity_input="input 0 images, state=0.05, red mug instruction",
                        reps_per_variant=args.reps, warmup_per_variant=args.warmup, graph_warmup=3,
                        cold_requests=[], parity_requests=[])
        reference_name = {"graph_dense": "native_dense", "graph_temporal": "eager_temporal",
                          "graph_conditioned_dense": "conditioned_dense",
                          "graph_conditioned_temporal": "conditioned_temporal"}
        references = {}

        def predict(name, input_id):
            images, state, instruction = inputs[input_id]
            with caches.get(name, nullcontext()) as cache:
                torch.cuda.synchronize()
                started = time.perf_counter()
                actions = policy.predict_action(images=images, state=state, instruction=instruction)
                torch.cuda.synchronize()
                seconds = time.perf_counter() - started
                stats = dict(cache.last_stats) if cache is not None else {}
            if not np.isfinite(actions).all():
                raise FloatingPointError(name)
            return seconds, actions, stats

        def difference(name, input_id, actions):
            original = "eager_temporal" if name.endswith("temporal") else "native_dense"
            reference = references[original, input_id]
            delta = actions.astype(np.float64) - reference.astype(np.float64)
            return dict(bitwise_prefactor=bool(np.array_equal(actions, reference)),
                        max_abs_prefactor=float(np.abs(delta).max()),
                        relative_l2_prefactor=float(np.linalg.norm(delta) / max(np.linalg.norm(reference), 1e-12)))

        for request, input_id in enumerate((*range(len(inputs)), 0)):
            for name in variants:
                seconds, actions, stats = predict(name, input_id)
                own = reference_name.get(name, name)
                key = (own, input_id)
                if own == name and key not in references:
                    references[key] = actions.copy()
                if not np.array_equal(actions, references[key]):
                    raise AssertionError(f"own-eager parity/request isolation failed: {name}, input {input_id}")
                row = dict(variant=name, request=request, input_id=input_id, **difference(name, input_id, actions))
                manifest["parity_requests"].append(row)
                if request == 0:
                    manifest["cold_requests"].append(dict(variant=name, seconds=seconds, cache=stats))
                write()
            print(f"request {request}, input {input_id}: own-eager parity passed for all eight variants", flush=True)
        for name, cache in caches.items():
            if name.startswith("graph_"):
                stats = cache.graph_stats()
                required = {"dense"}
                if name.endswith("temporal"):
                    required.add("action")
                if "conditioned" in name:
                    required.add("partial")
                if set(stats) != required or not all(row["captured"] for row in stats.values()):
                    raise AssertionError(f"missing graph capture: {name}")
        for name in variants:
            for rep in range(args.warmup):
                predict(name, rep % 3)
        samples = {name: [] for name in variants}
        with (out / "requests.jsonl").open("w", buffering=1) as raw:
            for rep in range(args.reps):
                order = variants[rep % len(variants):] + variants[:rep % len(variants)]
                for name in order:
                    seconds, actions, stats = predict(name, rep % 3)
                    own = reference_name.get(name, name)
                    if not np.array_equal(actions, references[own, rep % 3]):
                        raise AssertionError(f"timed own-eager parity failed: {name}")
                    if "conditioned" in name:
                        partials = 1 if name.endswith("temporal") else 9
                        if (stats["dense_video_steps"] != 1 or stats["partial_video_steps"] != partials or
                            stats["conditioned_token_layer_reuses"] != 98 * 30 * partials or
                            stats["action_layer_updates"] != 300):
                            raise AssertionError("unexpected conditioned/action computation count")
                    samples[name].append(seconds)
                    raw.write(json.dumps(dict(repeat=rep, order=list(order), variant=name, input_id=rep % 3,
                        seconds=seconds, bitwise_own_eager=True, cache=stats,
                        **difference(name, rep % 3, actions))) + "\n")
                print(f"repeat {rep}: " + " ".join(f"{name}={samples[name][-1]*1000:.2f}ms" for name in order), flush=True)
        if versions != {name: parameter._version for name, parameter in policy.model.named_parameters()}:
            raise AssertionError("parameter versions changed")
        metrics = {name: summary(values) for name, values in samples.items()}
        means = {name: values["mean"] for name, values in metrics.items()}
        report = dict(status="MEASURED", end_utc=datetime.now(timezone.utc).isoformat(), seconds=metrics,
            dense_factor_eager=means["native_dense"] / means["conditioned_dense"],
            dense_factor_graph=means["graph_dense"] / means["graph_conditioned_dense"],
            temporal_factor_eager=means["eager_temporal"] / means["conditioned_temporal"],
            temporal_factor_graph=means["graph_temporal"] / means["graph_conditioned_temporal"],
            temporal_vs_dense_graph_before=means["graph_dense"] / means["graph_temporal"],
            temporal_vs_dense_graph_after=means["graph_conditioned_dense"] / means["graph_conditioned_temporal"],
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
            graphs={name: cache.graph_stats() for name, cache in caches.items() if name.startswith("graph_")},
            bitwise_own_eager=True, all_bitwise_prefactor=all(row["bitwise_prefactor"] for row in manifest["parity_requests"]),
            gpu_after=inventory(), sr=None)
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
