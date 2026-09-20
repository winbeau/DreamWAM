#!/usr/bin/env python3
"""One common factor on strengthened Dense and fixed guided token sparsity.

Report complete requests with/without exact-prompt encoding memoization. Both
arms have transformer graphs; sparse remains r5/10%/guidance=1. Cache misses
and capture/setup are reported separately from repeated-instruction requests.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

from benchmark_ffn_context_cache import command, sha256, summary
from dreamwam.config import load_release_config
from dreamwam.policy import build_policy
from dreamwam.sparse.action_guided_visual_token_cache import ActionGuidedVisualTokenCache
from dreamwam.sparse.conditioned_frame_cache import ConditionedFrameCache, GraphedConditionedFrameCache
from dreamwam.sparse.prompt_cache import PromptEncodingCache
from dreamwam.sparse.visual_cache_graphs import GraphedVisualTokenCache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--config", default="configs/dreamwam_joint.yaml")
    parser.add_argument("--reps", type=int, default=24)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.reps < 12 or args.reps % 12:
        parser.error("reps must be a positive multiple of 12")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    inventory = lambda: command("nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,memory.used", "--format=csv")
    manifest = dict(status="RUNNING", start_utc=datetime.now(timezone.utc).isoformat(),
        argv=sys.argv, pid=os.getpid(), git=command("git", "rev-parse", "HEAD"),
        dirty=command("git", "status", "--porcelain"), python=platform.python_version(),
        torch=torch.__version__, cuda=torch.version.cuda, gpu_before=inventory(),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        sources={p: sha256(p) for p in [__file__, "dreamwam/sparse/prompt_cache.py",
            "dreamwam/components.py", "dreamwam/policy.py", "dreamwam/sparse/visual_cache_graphs.py",
            "dreamwam/sparse/conditioned_frame_cache.py", args.config]},
        input_manifest_sha256=sha256(args.inputs), factor="exact prompt encoding reuse on both arms",
        sparse=dict(refresh_every=5, token_keep_ratio=0.1, action_guidance_weight=1.0, graph_scope="all_transformers"),
        dense="conditioned-frame Dense with transformer graphs",
        latency_boundary="synchronized complete predict_action through CPU actions",
        environment="unchanged existing Python/torch; no installs or sync", sr=None,
        cold_requests=[], miss_requests=[], parity=[], timed_requests=0)
    # pathlib keys must be strings for JSON provenance.
    manifest["sources"] = {str(k): v for k, v in manifest["sources"].items()}
    write = lambda: (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write()
    graphs = {}; policy = None; original_encoder = None
    try:
        release = load_release_config(args.config)
        manifest["checkpoint_sha256"] = sha256(release.paths.checkpoint)
        data = json.loads(args.inputs.read_text())
        if data["status"] != "CAPTURED" or len(data["inputs"]) != 3:
            raise ValueError("require the three captured real observations")
        inputs = []
        for entry in data["inputs"]:
            p = args.inputs.parent / entry["path"]
            if sha256(p) != entry["sha256"]:
                raise ValueError("captured input changed")
            with np.load(p, allow_pickle=False) as values:
                inputs.append(dict(images={k: values[k].copy() for k in ("agentview", "wrist")},
                    state=values["state"].copy(), instruction=str(values["instruction"].item())))
        # Repeated prompt with different real images/state tests the cache boundary.
        inputs.append({**inputs[1], "instruction": inputs[0]["instruction"]})
        manifest["inputs"] = data["inputs"]
        manifest["extra_parity_input"] = "input 1 images/state with input 0 instruction; diagnostic only"
        policy = build_policy(release, device="cuda")
        evaluation = dict(policy.evaluation)
        if evaluation["denoising_steps"] != 10 or evaluation["action_horizon"] != 32:
            raise ValueError("retain the released sampler protocol")
        versions = {n: p._version for n, p in policy.model.named_parameters()}
        original_encoder = policy.text_encoder
        text = {arm: PromptEncodingCache(original_encoder) for arm in ("dense", "sparse")}
        eager = dict(dense=ConditionedFrameCache(policy.model, refresh_every=1),
            sparse=ActionGuidedVisualTokenCache(policy.model, refresh_every=5, keep_ratio=0.1, guidance_weight=1))
        graphs = dict(dense=GraphedConditionedFrameCache(policy.model, refresh_every=1),
            sparse=GraphedVisualTokenCache(policy.model, refresh_every=5, keep_ratio=0.1,
                                         guidance_weight=1, graph_partial=True))
        variants = ("dense_raw", "dense_cached", "sparse_raw", "sparse_cached")
        manifest.update(evaluation=evaluation, dtype=str(policy.dtype), variants=variants, reps=args.reps)

        def predict(name, input_id, *, use_graph=True):
            arm, encoding = name.split("_")
            policy.text_encoder = text[arm] if encoding == "cached" else original_encoder
            cache = graphs[arm] if use_graph else eager[arm]
            try:
                with cache:
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    actions = policy.predict_action(**inputs[input_id])
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - started
                    stats = dict(cache.last_stats)
                if not np.isfinite(actions).all():
                    raise FloatingPointError(name)
                return elapsed, actions, stats
            finally:
                policy.text_encoder = original_encoder

        refs = {}
        for input_id in range(4):
            for arm in ("dense", "sparse"):
                _, actions, _ = predict(arm + "_raw", input_id, use_graph=False)
                refs[arm, input_id] = actions.copy()
        # Direct real-encoder tensor checks, including cache output ownership.
        from dreamwam.preprocessing.libero import PROMPT_TEMPLATE
        for item in inputs[:3]:
            prompt = [PROMPT_TEMPLATE.format(task=item["instruction"])]
            expected = original_encoder(prompt)
            for arm in text:
                actual = text[arm](prompt)
                if not all(torch.equal(a, b) for a, b in zip(actual, expected)):
                    raise AssertionError("cached context/mask changed")
                actual[0].zero_(); actual[1].zero_()
                if not all(torch.equal(a, b) for a, b in zip(text[arm](prompt), expected)):
                    raise AssertionError("cache output aliased stored context")
        for request, input_id in enumerate((0, 1, 2, 3, 0)):
            for name in variants:
                seconds, actions, stats = predict(name, input_id)
                if not np.array_equal(actions, refs[name.split("_")[0], input_id]):
                    raise AssertionError(f"graph/cache eager parity failed: {name}/{input_id}")
                manifest["parity"].append(dict(variant=name, input_id=input_id, bitwise_own_uncached_eager=True))
                if request == 0:
                    manifest["cold_requests"].append(dict(variant=name, seconds=seconds, stats=stats))
                write()
        # Explicit misses with warm graphs; do not hide first-instruction cost.
        for input_id in range(3):
            for arm in text:
                text[arm].clear()
                seconds, actions, _ = predict(arm + "_cached", input_id)
                if text[arm].last_hit or not np.array_equal(actions, refs[arm, input_id]):
                    raise AssertionError("miss did not encode the expected prompt")
                manifest["miss_requests"].append(dict(arm=arm, input_id=input_id, seconds=seconds))
        for input_id in range(3):
            for name in variants:
                predict(name, input_id)
        samples = {name: [] for name in variants}
        with (args.out_dir / "requests.jsonl").open("w", buffering=1) as journal:
            for rep in range(args.reps):
                order = variants[rep % 4:] + variants[:rep % 4]
                for name in order:
                    seconds, actions, stats = predict(name, rep % 3)
                    arm, encoding = name.split("_")
                    if not np.array_equal(actions, refs[arm, rep % 3]):
                        raise AssertionError("timed output changed")
                    if encoding == "cached" and not text[arm].last_hit:
                        raise AssertionError("timed repeated prompt was not a cache hit")
                    if stats["action_layer_updates"] != 300:
                        raise AssertionError("action steps changed")
                    if arm == "sparse" and (stats["computed_video_token_layers"] != 9720 or
                            stats["partial_video_steps"] != 1 or stats["action_mass_builds"] != 1):
                        raise AssertionError("sparse executed budget changed")
                    samples[name].append(seconds)
                    journal.write(json.dumps(dict(repeat=rep, order=order, variant=name, input_id=rep % 3,
                        seconds=seconds, bitwise_own_uncached_eager=True, cache=stats,
                        prompt_cache=text[arm].stats() if encoding == "cached" else None)) + "\n")
                    manifest["timed_requests"] += 1
                write()
                print(f"timed round {rep + 1}/{args.reps}", flush=True)
        if evaluation != policy.evaluation or versions != {n: p._version for n, p in policy.model.named_parameters()}:
            raise AssertionError("weights or settings changed")
        metrics = {name: summary(values) for name, values in samples.items()}
        means = {name: value["mean"] for name, value in metrics.items()}
        result = dict(status="MEASURED", end_utc=datetime.now(timezone.utc).isoformat(), seconds=metrics,
            sparse_vs_matched_dense_before=means["dense_raw"] / means["sparse_raw"],
            sparse_vs_matched_dense_after=means["dense_cached"] / means["sparse_cached"],
            prompt_factor_dense=means["dense_raw"] / means["dense_cached"],
            prompt_factor_sparse=means["sparse_raw"] / means["sparse_cached"],
            graphs={k: v.graph_stats() for k, v in graphs.items()},
            prompt_caches={k: v.stats() for k, v in text.items()}, gpu_after=inventory(), sr=None,
            limitations=["Three exposed real scenes plus one input-boundary diagnostic",
                "Warm means assume a repeated instruction; six first-instruction misses are separate",
                "Sparse approximation unchanged, but complete official paired SR remains required"])
        (args.out_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        manifest.update(status="MEASURED", end_utc=result["end_utc"]); write()
        print(json.dumps(result, indent=2), flush=True)
    except BaseException as exc:
        manifest.update(status="ERROR", error=repr(exc), end_utc=datetime.now(timezone.utc).isoformat()); write(); raise
    finally:
        if policy is not None and original_encoder is not None:
            policy.text_encoder = original_encoder
        for cache in graphs.values():
            cache.close_graphs()


if __name__ == "__main__":
    main()
