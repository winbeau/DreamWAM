#!/usr/bin/env python3
"""Measure only transformer graph dispatch, with equally graphed Dense controls.

All timings encompass full predict_action and CPU actions. Cold calls, capture
costs and model loading are reported separately from warmed rotating samples.
Every buffered/graphed output must be bitwise its own eager variant on every
request. No compile, fast math, RNG, scheduler or scientific-setting changes.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

import numpy as np
import torch

from benchmark_ffn_context_cache import command, inputs_for, sha256, summary
from dreamwam.config import load_release_config
from dreamwam.policy import build_policy
from dreamwam.sparse.action_guided_visual_token_cache import ActionGuidedVisualTokenCache
from dreamwam.sparse.visual_cache_graphs import GraphedVisualTokenCache
from dreamwam.sparse.visual_step_cache import VisualStepCache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dreamwam_joint.yaml")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--graph-warmup", type=int, default=3)
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--include-partial-factor", action="store_true",
                        help="add matched Dense and guided variants with partial-refresh capture enabled")
    parser.add_argument("--include-temporal-control", action="store_true",
                        help="compare full visual refresh against 10 percent with all-transformer capture fixed")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    if min(args.warmup, args.graph_warmup) < 1 or args.reps < 2:
        parser.error("positive warmups and at least two repetitions required")
    if args.include_temporal_control and not args.include_partial_factor:
        parser.error("temporal budget comparison requires --include-partial-factor for matched graph scope")
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
                    sources={name: sha256(name) for name in (
                        "dreamwam/sparse/visual_cache_graphs.py", "dreamwam/sparse/visual_step_cache.py",
                        "dreamwam/sparse/visual_token_cache.py", "dreamwam/sparse/action_guided_visual_token_cache.py",
                        "scripts/sparse/benchmark_visual_cache_graphs.py", "pyproject.toml", "requirements.txt", args.config)},
                    factor=("visual refresh budget at fixed interval 5 and all-transformer graph dispatch"
                            if args.include_temporal_control else
                            "CUDA graph dispatch for dense Joint and cached-video action transformer calls"),
                    include_partial_factor=args.include_partial_factor,
                    include_temporal_control=args.include_temporal_control,
                    scope="selection and all policy/pre/post-DiT/RNG/scheduler logic remain eager; partial capture is a separate optional factor",
                    environment="existing .venv reused unchanged; no install or sync; uv.lock absent",
                    latency_boundary="synchronized full predict_action through CPU action output",
                    sr=None)
    write_manifest = lambda: (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_manifest()
    policy = None
    caches = {}
    try:
        started = time.perf_counter()
        policy = build_policy(release, device="cuda")
        manifest.update(model_load_seconds=time.perf_counter() - started,
                        evaluation=dict(policy.evaluation), dtype=str(policy.dtype),
                        setting=release.setting, sparse=policy.sparse_config.describe(),
                        fast_ops_enabled=policy.model.fast_ops_enabled)
        versions = {key: value._version for key, value in policy.model.named_parameters()}
        inputs, provenance = inputs_for(argparse.Namespace(inputs_npz=None), policy.image_size)
        timing_input_count = len(inputs)
        # An additional correctness-only request changes language and proprio too.
        inputs.append((inputs[0][0], np.full(8, 0.05, dtype=np.float32),
                       "pick up the red mug and place it in the basket"))
        manifest.update(inputs=provenance, extra_parity_input="input 0 images, state[8]=0.05, red mug instruction",
                        warmup_per_variant=args.warmup, graph_warmup=args.graph_warmup, reps_per_variant=args.reps)
        for label, keep, interval in (("dense", 1.0, 1), ("guided", 0.1, 5)):
            options = dict(keep_ratio=keep, refresh_every=interval, guidance_weight=1.0)
            caches[f"eager_{label}"] = ActionGuidedVisualTokenCache(policy.model, **options)
            for kind, enabled in (("buffered", False), ("graphed", True)):
                caches[f"{kind}_{label}"] = GraphedVisualTokenCache(
                    policy.model, **options, graph_enabled=enabled, graph_warmup=args.graph_warmup)
        variants = ("native_dense", "eager_dense", "eager_guided", "buffered_dense", "buffered_guided",
                    "graphed_dense", "graphed_guided")
        if args.include_partial_factor:
            for label, keep, interval in (("dense", 1.0, 1), ("guided", 0.1, 5)):
                name = f"graphed_partial_{label}"
                caches[name] = GraphedVisualTokenCache(
                    policy.model, keep_ratio=keep, refresh_every=interval, guidance_weight=1.0,
                    graph_warmup=args.graph_warmup, graph_partial=True)
                variants += (name,)
        if args.include_temporal_control:
            # The original V1 class is the eager reference used by the frozen SR
            # run. Full-budget graph sampling must reproduce that exact output.
            caches["eager_temporal"] = VisualStepCache(policy.model, refresh_every=5)
            caches["graphed_temporal"] = GraphedVisualTokenCache(
                policy.model, keep_ratio=1.0, refresh_every=5, guidance_weight=1.0,
                graph_warmup=args.graph_warmup, graph_partial=True)
            variants += ("eager_temporal", "graphed_temporal")
        manifest["variant_options"] = {
            name: dict(refresh_every=cache.refresh_every, token_keep_ratio=cache.keep_ratio,
                       action_guidance_weight=getattr(cache, "guidance_weight", 0.0),
                       graph_enabled=getattr(cache, "graph_enabled", False),
                       graph_partial=getattr(cache, "graph_partial", False))
            for name, cache in caches.items()}

        def reference_for(variant):
            if variant.endswith("temporal"):
                return "eager_temporal"
            return "eager_guided" if variant.endswith("guided") else "native_dense"

        def predict(variant, input_id):
            images, state, instruction = inputs[input_id]
            with caches.get(variant, nullcontext()) as cache:
                torch.cuda.synchronize()
                started = time.perf_counter()
                actions = policy.predict_action(images=images, state=state, instruction=instruction)
                torch.cuda.synchronize()
                seconds = time.perf_counter() - started
                stats = {} if cache is None else dict(cache.last_stats)
            if not np.isfinite(actions).all():
                raise FloatingPointError(f"non-finite {variant} output")
            return seconds, actions, stats

        references = {}
        manifest["cold_requests"] = []
        # The first call of every variant, including capture, is kept in evidence.
        # Follow with changed inputs and a return to input 0 to detect stale buffers.
        for input_id in (*range(len(inputs)), 0):
            for variant in variants:
                seconds, actual, stats = predict(variant, input_id)
                if variant in ("native_dense", "eager_guided", "eager_temporal"):
                    key = (variant, input_id)
                    if key in references and not np.array_equal(actual, references[key]):
                        raise AssertionError(f"eager request isolation failed: {key}")
                    references[key] = actual.copy()
                reference_variant = reference_for(variant)
                if not np.array_equal(actual, references[(reference_variant, input_id)]):
                    raise AssertionError(f"bitwise eager parity failed: {variant}, input {input_id}")
                if input_id == 0 and len(manifest["cold_requests"]) < len(variants):
                    manifest["cold_requests"].append(dict(variant=variant, seconds=seconds, cache=stats))
                    write_manifest()
            print(f"parity input {input_id}: all {len(variants)} variants passed", flush=True)
        manifest["bitwise_eager_parity"] = True
        for name in (key for key in variants if key.startswith("graphed")):
            graph_stats = caches[name].graph_stats()
            required = {"dense", "action"} if name.endswith(("guided", "temporal")) else {"dense"}
            if name == "graphed_partial_guided":
                required.add("partial")
            if set(graph_stats) != required or not all(value["captured"] for value in graph_stats.values()):
                raise AssertionError(f"required CUDA graph missing: {name}")
        for variant in variants:
            for index in range(args.warmup):
                predict(variant, index % timing_input_count)
        write_manifest()
        samples = {name: [] for name in variants}
        with (out / "requests.jsonl").open("w", buffering=1) as raw:
            for repeat in range(args.reps):
                order = variants[repeat % len(variants):] + variants[:repeat % len(variants)]
                input_id = repeat % timing_input_count
                for variant in order:
                    seconds, actual, stats = predict(variant, input_id)
                    reference_variant = reference_for(variant)
                    if not np.array_equal(actual, references[(reference_variant, input_id)]):
                        raise AssertionError(f"timed bitwise parity failed: {variant}, input {input_id}")
                    difference = actual.astype(np.float64) - references[("native_dense", input_id)].astype(np.float64)
                    row = dict(repeat=repeat, order=list(order), variant=variant, input_id=input_id,
                               seconds=seconds, bitwise_own_eager=True, cache=stats,
                               action_relative_l2_vs_dense=float(np.linalg.norm(difference) /
                                   max(np.linalg.norm(references[("native_dense", input_id)]), 1e-12)))
                    raw.write(json.dumps(row) + "\n")
                    samples[variant].append(seconds)
                print(f"repeat {repeat:02d}: " + " ".join(f"{name}={samples[name][-1]*1000:.2f}ms" for name in order), flush=True)
        if versions != {key: value._version for key, value in policy.model.named_parameters()}:
            raise AssertionError("model parameter versions changed")
        mean = lambda name: statistics.fmean(samples[name])
        report = dict(status="MEASURED", end_utc=datetime.now(timezone.utc).isoformat(),
                      latency_seconds={name: summary(values) for name, values in samples.items()},
                      speedups=dict(
                          guided_vs_matched_eager=mean("eager_dense") / mean("eager_guided"),
                          guided_vs_matched_graphed=mean("graphed_dense") / mean("graphed_guided"),
                          graph_factor_dense=mean("eager_dense") / mean("graphed_dense"),
                          graph_factor_guided=mean("eager_guided") / mean("graphed_guided"),
                          graph_vs_buffered_dense=mean("buffered_dense") / mean("graphed_dense"),
                          graph_vs_buffered_guided=mean("buffered_guided") / mean("graphed_guided"),
                          graphed_guided_vs_native=mean("native_dense") / mean("graphed_guided")),
                      graphs={name: cache.graph_stats() for name, cache in caches.items()
                              if isinstance(cache, GraphedVisualTokenCache)},
                      bitwise_own_eager_every_sample=True, unchanged_parameters=True, sr=None)
        if args.include_partial_factor:
            report["speedups"].update(
                partial_factor_guided=mean("graphed_guided") / mean("graphed_partial_guided"),
                partial_factor_dense_control=mean("graphed_dense") / mean("graphed_partial_dense"),
                guided_vs_matched_all_graphed=mean("graphed_partial_dense") / mean("graphed_partial_guided"),
                all_graphed_guided_vs_native=mean("native_dense") / mean("graphed_partial_guided"))
        if args.include_temporal_control:
            report["speedups"].update(
                temporal_vs_matched_eager=mean("eager_dense") / mean("eager_temporal"),
                temporal_vs_matched_graphed=mean("graphed_partial_dense") / mean("graphed_temporal"),
                graph_factor_temporal=mean("eager_temporal") / mean("graphed_temporal"),
                token_budget_factor_graphed=mean("graphed_temporal") / mean("graphed_partial_guided"),
                all_graphed_temporal_vs_native=mean("native_dense") / mean("graphed_temporal"))
        (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        manifest.update(status="MEASURED", exit_code=0, end_utc=report["end_utc"], gpu_after=inventory(),
                        peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                        peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20)
        write_manifest()
        print(json.dumps(report, indent=2), flush=True)
    except Exception as error:
        manifest.update(status="FAILED", end_utc=datetime.now(timezone.utc).isoformat(),
                        error=f"{type(error).__name__}: {error}", gpu_after=inventory(),
                        graphs={name: cache.graph_stats() for name, cache in caches.items()
                                if isinstance(cache, GraphedVisualTokenCache)})
        write_manifest()
        raise
    finally:
        for cache in caches.values():
            if isinstance(cache, GraphedVisualTokenCache):
                cache.close_graphs()
        if policy is not None:
            policy.close()


if __name__ == "__main__":
    main()
