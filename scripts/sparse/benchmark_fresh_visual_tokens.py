#!/usr/bin/env python3
"""Every-step token sparsity: current AV versus equal-budget uniform selection.

Four matched, rotating arms on three real inputs. Native full-compute Dense and
the stronger invariant-frame Dense are both included; all share exact prompt
memoization and transformer CUDA graphs. No new sampler or precision change.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from benchmark_ffn_context_cache import command, sha256, summary
from dreamwam.config import load_release_config
from dreamwam.policy import build_policy
from dreamwam.sparse.conditioned_frame_cache import ConditionedFrameCache, GraphedConditionedFrameCache
from dreamwam.sparse.fresh_visual_tokens import FreshVisualTokenSparsity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--config", default="configs/dreamwam_joint.yaml")
    parser.add_argument("--reps", type=int, default=24)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.reps < 12 or args.reps % 12:
        parser.error("reps must be a multiple of 12")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    manifest = dict(status="RUNNING", start_utc=datetime.now(timezone.utc).isoformat(),
        argv=sys.argv, pid=os.getpid(), git=command("git", "rev-parse", "HEAD"),
        dirty=command("git", "status", "--porcelain"), torch=torch.__version__,
        cuda=torch.version.cuda, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        input_manifest_sha256=sha256(args.inputs),
        sources={p: sha256(p) for p in [__file__, "dreamwam/sparse/fresh_visual_tokens.py",
            "dreamwam/policy.py", "dreamwam/model.py", "dreamwam/mot.py", "pyproject.toml", "requirements.txt"]},
        latency_boundary="synchronized complete predict_action through CPU actions",
        semantics="10% current visual rows each step including step 0; inactive current-input bypass; no visual cache",
        sr=None, parity=[], setup_requests=[], misses=[], timed_requests=0)
    write = lambda: (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write()
    runtimes = {}
    policy = None
    try:
        release = load_release_config(args.config)
        manifest["checkpoint_sha256"] = sha256(release.paths.checkpoint)
        captured = json.loads(args.inputs.read_text())
        if captured["status"] != "CAPTURED" or len(captured["inputs"]) != 3:
            raise ValueError("require three captured real observations")
        inputs = []
        for item in captured["inputs"]:
            path = args.inputs.parent / item["path"]
            if sha256(path) != item["sha256"]:
                raise ValueError("input hash changed")
            with np.load(path, allow_pickle=False) as data:
                inputs.append(dict(images={k: data[k].copy() for k in ("agentview", "wrist")},
                    state=data["state"].copy(), instruction=str(data["instruction"].item())))
        inputs.append({**inputs[1], "instruction": inputs[0]["instruction"]})
        manifest["inputs"] = captured["inputs"]
        started = time.perf_counter()
        policy = build_policy(release, device="cuda", prompt_cache={"capacity": 8})
        manifest["load_seconds"] = time.perf_counter() - started
        settings = dict(policy.evaluation)
        assert settings["denoising_steps"] == 10 and settings["action_horizon"] == 32
        versions = {n: p._version for n, p in policy.model.named_parameters()}
        manifest.update(evaluation=settings, dtype=str(policy.dtype), reps=args.reps)
        eager = dict(dense=FreshVisualTokenSparsity(policy.model, keep_ratio=1),
            dense_strong=ConditionedFrameCache(policy.model, refresh_every=1),
            uniform10=FreshVisualTokenSparsity(policy.model, keep_ratio=0.1, selection="uniform"),
            action10=FreshVisualTokenSparsity(policy.model, keep_ratio=0.1, selection="action"))
        runtimes = dict(dense=FreshVisualTokenSparsity(policy.model, keep_ratio=1, graph_enabled=True),
            dense_strong=GraphedConditionedFrameCache(policy.model, refresh_every=1),
            uniform10=FreshVisualTokenSparsity(policy.model, keep_ratio=0.1, selection="uniform", graph_enabled=True),
            action10=FreshVisualTokenSparsity(policy.model, keep_ratio=0.1, selection="action", graph_enabled=True))
        variants = tuple(runtimes)
        manifest["variants"] = variants

        def predict(name, input_id, graph=True):
            runtime = runtimes[name] if graph else eager[name]
            with runtime:
                torch.cuda.synchronize()
                start = time.perf_counter()
                actions = policy.predict_action(**inputs[input_id])
                torch.cuda.synchronize()
                seconds = time.perf_counter() - start
                stats = dict(runtime.last_stats)
            if not np.isfinite(actions).all():
                raise FloatingPointError(name)
            return seconds, actions, stats

        refs = {}
        arrays = {}
        prompt = policy._prompt_cache_runtime
        policy.text_encoder = prompt.encoder
        try:
            for i in range(4):
                for name in variants:
                    _, actions, _ = predict(name, i, graph=False)
                    refs[name, i] = actions.copy()
                    arrays[f"{name}_{i}"] = actions.copy()
                if not np.array_equal(refs["dense", i], refs["dense_strong", i]):
                    raise AssertionError("stronger Dense changes reference actions")
        finally:
            policy.text_encoder = prompt
        np.savez_compressed(args.out_dir / "reference-actions.npz", **arrays)
        manifest["reference_actions_sha256"] = sha256(args.out_dir / "reference-actions.npz")
        for call, i in enumerate((0, 1, 2, 3, 0)):
            for name in variants:
                seconds, actions, stats = predict(name, i)
                if not np.array_equal(actions, refs[name, i]):
                    raise AssertionError(f"graph/eager parity failed {name}/{i}")
                manifest["parity"].append(dict(variant=name, input_id=i, bitwise_own_uncached_eager=True))
                if call == 0:
                    manifest["setup_requests"].append(dict(variant=name, seconds=seconds, stats=stats))
                write()
        for i in range(3):
            for name in ("dense_strong", "action10"):
                prompt.clear()
                seconds, actions, _ = predict(name, i)
                if prompt.last_hit or not np.array_equal(actions, refs[name, i]):
                    raise AssertionError("first instruction miss parity failed")
                manifest["misses"].append(dict(variant=name, input_id=i, seconds=seconds))
        for i in range(3):
            for name in variants:
                predict(name, i)
        samples = {name: [] for name in variants}
        with (args.out_dir / "requests.jsonl").open("w", buffering=1) as journal:
            for rep in range(args.reps):
                order = variants[rep % 4:] + variants[:rep % 4]
                i = rep % 3
                for name in order:
                    seconds, actions, stats = predict(name, i)
                    if not np.array_equal(actions, refs[name, i]) or not prompt.last_hit:
                        raise AssertionError("timed output or warm prompt boundary changed")
                    expected = 61740 if name == "dense_strong" else (88200 if name == "dense" else 9000)
                    if stats["action_layer_updates"] != 300 or stats["computed_video_token_layers"] != expected:
                        raise AssertionError("executed transformer budget changed")
                    if name in ("uniform10", "action10"):
                        if stats["denoising_steps"] != 10 or stats["selection_builds"] != 10 or stats["reused_visual_steps"]:
                            raise AssertionError("a step was skipped, dense, or cached")
                        if any(len(row) != 30 for row in stats["selected_indices"]):
                            raise AssertionError("token budget changed")
                    relative_l2 = float(np.linalg.norm(actions - refs["dense", i]) / max(np.linalg.norm(refs["dense", i]), 1e-12))
                    samples[name].append(seconds)
                    journal.write(json.dumps(dict(repeat=rep, order=order, variant=name, input_id=i,
                        seconds=seconds, bitwise_own_uncached_eager=True, counters=stats,
                        action_relative_l2_vs_dense=relative_l2)) + "\n")
                    manifest["timed_requests"] += 1
                write()
                print(f"timed round {rep + 1}/{args.reps}", flush=True)
        if settings != policy.evaluation or versions != {n: p._version for n, p in policy.model.named_parameters()}:
            raise AssertionError("weights or settings changed")
        metrics = {name: summary(values) for name, values in samples.items()}
        result = dict(status="MEASURED", end_utc=datetime.now(timezone.utc).isoformat(), seconds=metrics,
            speedups={name: metrics["dense_strong"]["mean"] / metrics[name]["mean"] for name in variants},
            versus_full_recompute={name: metrics["dense"]["mean"] / metrics[name]["mean"] for name in variants},
            graphs={name: runtime.graph_stats() for name, runtime in runtimes.items()}, sr=None,
            limitations=["three exposed real calibration observations; no SR inference",
                "warm repeated instructions; first instruction misses and setup reported separately",
                "inactive visual rows bypass current transformer input; AV keys are also pruned",
                "frame quotas add structural support but do not implement full AV-VV context bridging"])
        (args.out_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        manifest.update(status="MEASURED", end_utc=result["end_utc"])
        write()
        print(json.dumps(result, indent=2), flush=True)
    except BaseException as exc:
        manifest.update(status="ERROR", error=repr(exc), end_utc=datetime.now(timezone.utc).isoformat())
        write()
        raise
    finally:
        for runtime in runtimes.values():
            runtime.close_graphs()
        if policy is not None:
            policy.close()


if __name__ == "__main__":
    main()
