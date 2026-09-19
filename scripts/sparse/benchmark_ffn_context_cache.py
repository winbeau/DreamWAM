#!/usr/bin/env python3
"""Matched full-request measurement of ONE factor: visual FFN recompute ratio.

Every request calls the released policy, including text/VAE encoding, all ten
denoising steps, cache anchors/selection/reconstruction, and CPU action output.
Native Dense, the same wrapper at full budget (matched Dense), and one selected
ratio alternate on the same GPU, checkpoint, inputs, and released RNG settings.
Synthetic inputs are explicitly labelled and are not closed-loop SR evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from dreamwam.config import load_release_config
from dreamwam.policy import build_policy
from dreamwam.sparse.ffn_context_cache import VisualFFNContextCache
from dreamwam.sparse.ffn_neuron_cache import VisualFFNNeuronCache


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command(*args):
    result = subprocess.run(args, capture_output=True, text=True)
    return dict(exit_code=result.returncode, stdout=result.stdout.strip(), stderr=result.stderr.strip())


def summary(values):
    return dict(samples=len(values), mean=statistics.fmean(values), p50=float(np.percentile(values, 50)),
                p95=float(np.percentile(values, 95)), min=min(values), max=max(values),
                stdev=statistics.stdev(values) if len(values) > 1 else 0.0)


def inputs_for(args, image_size):
    if args.inputs_npz:
        inputs = []
        for name in args.inputs_npz:
            with np.load(name, allow_pickle=False) as data:
                inputs.append((dict(agentview=data["agentview"], wrist=data["wrist"]),
                               data["state"], str(data["instruction"].item())))
        return inputs, [dict(path=str(Path(name).resolve()), sha256=sha256(name)) for name in args.inputs_npz]
    inputs = []
    for seed in (0, 1, 2):
        rng = np.random.default_rng(seed)
        images = {key: rng.integers(0, 255, (image_size, image_size, 3), dtype=np.uint8)
                  for key in ("agentview", "wrist")}
        inputs.append((images, np.zeros(8, dtype=np.float32), "pick up the black bowl and place it on the plate"))
    return inputs, dict(kind="synthetic random images and zero proprio; latency screening only", seeds=[0, 1, 2])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dreamwam_joint.yaml")
    parser.add_argument("--keep-ratio", type=float, default=0.1)
    parser.add_argument("--cache-kind", choices=("tokens", "neurons"), default="tokens")
    parser.add_argument("--group-size", type=int, default=10, help="neuron-mask group size; dense anchor at each group start")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--reps", type=int, default=12)
    parser.add_argument("--inputs-npz", action="append")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    if not 0 <= args.keep_ratio < 1 or args.reps < 2 or args.warmup < 1:
        parser.error("require 0 <= keep-ratio < 1, reps >= 2, warmup >= 1")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    release = load_release_config(args.config)
    manifest = dict(status="RUNNING", start_utc=datetime.now(timezone.utc).isoformat(),
                    argv=sys.argv, cwd=str(Path.cwd()), pid=os.getpid(),
                    git=command("git", "rev-parse", "HEAD"), dirty=command("git", "status", "--porcelain"),
                    python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                    cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                    gpu_before=command("nvidia-smi", "--query-gpu=index,uuid,name,utilization.gpu,memory.used", "--format=csv"),
                    processes_before=command("nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv"),
                    checkpoint=dict(path=str(release.paths.checkpoint), sha256=sha256(release.paths.checkpoint)),
                    source_sha256={name: sha256(name) for name in (
                        "dreamwam/sparse/ffn_context_cache.py", "dreamwam/sparse/ffn_neuron_cache.py", "scripts/sparse/benchmark_ffn_context_cache.py",
                        "pyproject.toml", "requirements.txt", args.config)},
                    environment="existing .venv; no install/sync or dependency changes; model uv.lock absent",
                    factor=dict(name=f"visual FFN {args.cache_kind} recompute fraction", keep_ratio=args.keep_ratio,
                                cache_kind=args.cache_kind,
                                group_size=args.group_size if args.cache_kind == "neurons" else None,
                                reference="per-row latest computed input/output" if args.cache_kind == "tokens" else "dense current-request group anchor",
                                first_step="dense for every layer", action_guidance=False),
                    latency_boundary="full predict_action, synchronized wall clock, includes CPU action output",
                    sr=None)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    started = time.perf_counter()
    policy = build_policy(release, device="cuda")
    manifest["model_load_seconds"] = time.perf_counter() - started
    manifest["evaluation"] = dict(policy.evaluation)
    manifest["setting"] = release.setting
    manifest["sparse"] = policy.sparse_config.describe()
    manifest["dtype"] = str(policy.dtype)
    inputs, provenance = inputs_for(args, policy.image_size)
    manifest["inputs"] = provenance
    manifest["warmup_per_variant"] = args.warmup
    manifest["reps_per_variant"] = args.reps
    manifest["parameter_versions"] = {name: parameter._version for name, parameter in policy.model.named_parameters()}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    candidate = "context_cache" if args.cache_kind == "tokens" else "neuron_cache"
    cache_class = VisualFFNContextCache if args.cache_kind == "tokens" else VisualFFNNeuronCache
    extra = {} if args.cache_kind == "tokens" else {"group_size": args.group_size}
    caches = {"matched_dense": cache_class(policy.model, keep_ratio=1.0, **extra),
              candidate: cache_class(policy.model, keep_ratio=args.keep_ratio, **extra)}
    if args.cache_kind == "neurons":
        caches[candidate].prepare()
        manifest["static_weight_norm_setup_seconds"] = caches[candidate].setup_seconds
    variants = ("native_dense", "matched_dense", candidate)
    dense_actions = {}
    samples = {name: [] for name in variants}
    comparisons = []

    def predict(variant, input_id):
        manager = nullcontext(None) if variant == "native_dense" else caches[variant]
        images, state, instruction = inputs[input_id]
        with manager as cache:
            torch.cuda.synchronize()
            start = time.perf_counter()
            action = policy.predict_action(images=images, state=state, instruction=instruction)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            stats = {} if cache is None else cache.last_stats
        if not np.isfinite(action).all():
            raise FloatingPointError(f"non-finite action in {variant}")
        return elapsed, action, stats

    # The full-budget wrapper must reproduce native actions for every input.
    for input_id in range(len(inputs)):
        _, expected, _ = predict("native_dense", input_id)
        dense_actions[input_id] = expected
        _, actual, _ = predict("matched_dense", input_id)
        if not np.array_equal(actual, expected):
            raise AssertionError(f"full-budget parity failed for input {input_id}")
    manifest["full_budget_bitwise_parity"] = True
    for variant in variants:
        for index in range(args.warmup):
            predict(variant, index % len(inputs))

    with (out / "requests.jsonl").open("w", buffering=1) as raw:
        for repeat in range(args.reps):
            # Rotate order to balance GPU clock and host-load drift.
            order = variants[repeat % 3:] + variants[:repeat % 3]
            input_id = repeat % len(inputs)
            for variant in order:
                elapsed, actual, stats = predict(variant, input_id)
                expected = dense_actions[input_id]
                if variant != candidate and not np.array_equal(actual, expected):
                    raise AssertionError(f"Dense/request isolation parity failed: {variant} input {input_id}")
                difference = actual.astype(np.float64) - expected.astype(np.float64)
                row = dict(repeat=repeat, order=list(order), variant=variant, input_id=input_id,
                           seconds=elapsed, cache=stats,
                           action_max_abs=float(np.abs(difference).max()),
                           action_rmse=float(np.sqrt(np.mean(difference ** 2))),
                           action_relative_l2=float(np.linalg.norm(difference) / max(np.linalg.norm(expected), 1e-12)),
                           gripper_disagreements=int(np.count_nonzero(actual[:, -1] != expected[:, -1])))
                samples[variant].append(elapsed)
                comparisons.append(row)
                raw.write(json.dumps(row) + "\n")
                print(f"{repeat:02d} {variant:14s} {elapsed:.6f}s {stats}", flush=True)

    versions = {name: parameter._version for name, parameter in policy.model.named_parameters()}
    if versions != manifest["parameter_versions"]:
        raise AssertionError("model parameter version changed")
    result = dict(status="MEASURED", end_utc=datetime.now(timezone.utc).isoformat(),
                  latency_seconds={name: summary(values) for name, values in samples.items()},
                  speedup_vs_matched_mean=statistics.fmean(samples["matched_dense"]) / statistics.fmean(samples[candidate]),
                  speedup_vs_matched_p50=statistics.median(samples["matched_dense"]) / statistics.median(samples[candidate]),
                  native_vs_matched_mean=statistics.fmean(samples["native_dense"]) / statistics.fmean(samples["matched_dense"]),
                  full_budget_bitwise_parity=True, unchanged_parameter_versions=True,
                  max_action_relative_l2=max(row["action_relative_l2"] for row in comparisons if row["variant"] == candidate),
                  gpu_after=command("nvidia-smi", "--query-gpu=index,uuid,name,utilization.gpu,memory.used", "--format=csv"),
                  processes_after=command("nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv"),
                  sr=None, limitations=["No closed-loop SR; action error is a diagnostic, not a success criterion.",
                                       f"Only {args.cache_kind} reuse enabled; no action guidance, attention restriction, or other cache factor."])
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    manifest.update(status="MEASURED", exit_code=0, end_utc=result["end_utc"], full_budget_bitwise_parity=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
