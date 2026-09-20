#!/usr/bin/env python3
"""Audit closed-loop per-call timings; keep setup, model time and simulator time separate.

This does not judge success. Use paired_sr.py for benchmark-owned outcomes.
Closed-loop trajectories diverge: latency ratios here are descriptive, not
same-observation causal performance measurements.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from paired_sr import load_manifest


def distribution(values):
    if not values:
        return None
    ordered = sorted(values)
    def quantile(p):
        at = (len(ordered) - 1) * p
        low, high = math.floor(at), math.ceil(at)
        return ordered[low] + (ordered[high] - ordered[low]) * (at - low)
    return dict(samples=len(values), mean=statistics.fmean(values),
                p50=quantile(0.5), p95=quantile(0.95))


def inspect_run(directory):
    manifest, planned = load_manifest(directory)
    episodes = {}
    all_model, all_transport, first_model, warm_model, warm_transport = [], [], [], [], []
    for path in sorted(directory.glob("episodes/*/attempts/*/result.json")):
        result = json.loads(path.read_text())
        if result["status"] not in ("succeeded", "failed"):
            continue
        identity = tuple(result[k] for k in ("task_id", "init_index", "repeat", "seed"))
        if identity not in planned or identity in episodes:
            raise ValueError("unexpected or multiply settled episode identity")
        calls_path = directory / result["artifacts"]["policy_calls"]
        calls = json.loads(calls_path.read_text())
        if len(calls) != result["policy_calls"] or not calls:
            raise ValueError("per-call timing coverage differs from the episode record")
        model, transport, warm, warm_ipc = [], [], [], []
        for index, call in enumerate(calls, 1):
            if call["call_index"] != index:
                raise ValueError("per-call order changed")
            metadata = call["metadata"]
            seconds = metadata["timing"]["predict_seconds"]
            rpc = call["transport_seconds"]
            if any(type(t) not in (int, float) or not math.isfinite(t) or t <= 0 for t in (seconds, rpc)):
                raise ValueError("invalid recorded timing")
            model.append(seconds)
            transport.append(rpc)
            # Fixed before looking at outcomes: omit the first call of EACH episode.
            if index > 1 and metadata["diagnostics"].get("prompt_cache", {}).get("last_hit") is True:
                warm.append(seconds)
                warm_ipc.append(rpc)
        if not math.isclose(sum(transport), result["policy_seconds"], abs_tol=1e-5):
            raise ValueError("transport timing sum differs from episode policy_seconds")
        episodes[identity] = dict(episode_id=result["episode_id"], status=result["status"],
            policy_calls=len(calls), first_model_seconds=model[0],
            warm_model=distribution(warm), all_model=distribution(model),
            wall_seconds=result["wall_seconds"], env_seconds=result["env_seconds"],
            calls_sha256=hashlib.sha256(calls_path.read_bytes()).hexdigest(),
            result_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        all_model.extend(model)
        all_transport.extend(transport)
        first_model.append(model[0])
        warm_model.extend(warm)
        warm_transport.extend(warm_ipc)
    return manifest, planned, episodes, dict(
        coverage=dict(completed=len(episodes), planned=len(planned), complete=set(episodes) == planned),
        model_seconds_all=distribution(all_model), transport_seconds_all=distribution(all_transport),
        model_seconds_first_per_episode=distribution(first_model),
        model_seconds_warm=distribution(warm_model), transport_seconds_warm=distribution(warm_transport),
        episodes=[episodes[key] for key in sorted(episodes)])


def report(dense_dir, sparse_dir):
    dm, dp, de, dense = inspect_run(dense_dir)
    sm, sp, se, sparse = inspect_run(sparse_dir)
    if dm["benchmark"] != sm["benchmark"] or dm["protocol"] != sm["protocol"] or dp != sp:
        raise ValueError("different paired benchmark/protocol/episode manifests")
    complete = dense["coverage"]["complete"] and sparse["coverage"]["complete"]
    def ratio(key):
        a, b = dense[key], sparse[key]
        return a["mean"] / b["mean"] if complete and a and b else None
    pairable = complete and all(de[k]["warm_model"] and se[k]["warm_model"] for k in dp)
    balanced = (statistics.fmean(de[k]["warm_model"]["mean"] for k in dp)
                / statistics.fmean(se[k]["warm_model"]["mean"] for k in sp)) if pairable else None
    return dict(complete=complete, dense=dense, sparse=sparse,
        descriptive_warm_model_speedup=ratio("model_seconds_warm"),
        descriptive_warm_transport_speedup=ratio("transport_seconds_warm"),
        descriptive_all_model_speedup=ratio("model_seconds_all"),
        descriptive_episode_balanced_warm_model_speedup=balanced,
        warm_definition="call_index > 1 within each episode AND recorded prompt-cache hit",
        limitations=["Closed-loop observations, lengths and success outcomes may differ.",
            "First-call costs include prompt misses and any lazy graph capture.",
            "Model timing excludes IPC, model load, simulator and rendering.",
            "Use paired_sr.py for success and same-input replay for controlled speed measurements."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense", type=Path, required=True)
    parser.add_argument("--sparse", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = report(args.dense, args.sparse)
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with args.out.open("x") as handle:
        handle.write(text)
    print(text)


if __name__ == "__main__":
    main()
