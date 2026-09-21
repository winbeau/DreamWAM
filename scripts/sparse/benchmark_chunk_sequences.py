#!/usr/bin/env python3
"""Balanced complete-history adapter timings; M1 resets only at episode boundaries."""

import argparse
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch

from benchmark_ffn_context_cache import command
from benchmark_hybrid_schedules import diagnostics
from observation_sequences import load_sequences, sha256
from evaluation.action_eval.infer import create_policy
from dreamwam.sparse.chunk_budget import ChunkBudgetConfig, ObservationBudget
from dreamwam.sparse.conditioned_frame_cache import ConditionedFrameCache
from dreamwam.sparse.hybrid.config import HybridConfig
from dreamwam.sparse.hybrid.runtime import HybridVisualRuntime


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--options", type=Path, action="append", required=True)
    p.add_argument("--reps", type=int, default=4)
    p.add_argument("--authorized-gpus", type=int, nargs="+", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        p.error("select exactly one authorized CUDA GPU")
    if command("git", "status", "--porcelain")["stdout"]:
        p.error("require committed clean source")
    options = {path.stem: json.loads(path.read_text()) for path in args.options}
    if len(options) != len(args.options) or "dense" in options or args.reps < 1 or args.reps % (len(options) + 1):
        p.error("unique candidate names and repetitions divisible by total arm count are required")
    configs, budgets = {}, {}
    for name, option in options.items():
        if set(option) != {"prompt_cache", "hybrid_visual", "chunk_budget"} or option["prompt_cache"] != {"capacity": 8}:
            p.error("candidates must use the common prompt cache and explicit M1/M2/M3 options")
        configs[name] = HybridConfig.from_mapping(option["hybrid_visual"])
        budgets[name] = ChunkBudgetConfig.from_mapping(option["chunk_budget"])
        budgets[name].hybrid_configs(configs[name])
        if configs[name].diagnostics != "counters" or configs[name].backend != "cuda_graph":
            p.error("timings require uninstrumented CUDA graph candidates")
    manifest, sequences = load_sequences(args.inputs)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    report = dict(status="RUNNING", start_utc=datetime.now(timezone.utc).isoformat(), argv=sys.argv,
        commit=command("git", "rev-parse", "HEAD")["stdout"], inputs_sha256=sha256(args.inputs),
        input_split_sha256=manifest["episode_split_sha256"], options=options,
        options_sha256={path.stem: sha256(path) for path in args.options},
        torch=torch.__version__, cuda=torch.version.cuda, gpu=visible, reps=args.reps,
        resource_checks=[], references={}, setup=[], summary={}, sr=None,
        limitations=["complete-history replay on recorded development inputs, not closed-loop SR",
            "warm timings exclude episode-first instruction misses and setup; cold samples reported separately",
            "order balanced by complete trajectory; shared host, same-GPU external activity checked between trajectories",
            "same checkpoint is resident for all variants; each variant has its own graph cache",
            "attention probes and M1/M3 decisions are included in complete adapter prediction time"])
    def save():
        (args.out_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    def resources(phase):
        inventory = command("nvidia-smi", "-i", visible, "--query-gpu=uuid,memory.used,utilization.gpu", "--format=csv,noheader")
        proc = command("nvidia-smi", "-i", visible, "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader")
        foreign = [line for line in proc["stdout"].splitlines() if line.split(",")[0].strip() != str(os.getpid())]
        total_used = int(inventory["stdout"].split(",")[1].split()[0]) if not inventory["exit_code"] else 0
        own_used = sum(int(line.split(",")[1].split()[0]) for line in proc["stdout"].splitlines()
                       if line.split(",")[0].strip() == str(os.getpid()))
        # Some container-external allocations have no visible process entry.
        unexplained_memory = total_used - own_used
        report["resource_checks"].append(dict(phase=phase, utc=datetime.now(timezone.utc).isoformat(),
            inventory=inventory, processes=proc, foreign=foreign, unexplained_mib=unexplained_memory))
        if phase == "admission":
            host = command("nvidia-smi", "--query-gpu=index,uuid,memory.free,utilization.gpu", "--format=csv,noheader,nounits")
            cards = [[v.strip() for v in line.split(",")] for line in host["stdout"].splitlines()]
            target = inventory["stdout"].split(",")[0].strip()
            authorized = [c for c in cards if int(c[0]) in args.authorized_gpus]
            spares = [int(c[0]) for c in authorized if c[1] != target and int(c[2]) >= 35000 and int(c[3]) <= 20]
            report["admission"] = dict(host=host, authorized_gpus=args.authorized_gpus, unused_spares=spares)
            if not any(c[1] == target for c in authorized) or not spares or total_used > 32:
                raise RuntimeError("require an authorized empty policy GPU and an unused available spare")
        save()
        if proc["exit_code"] or inventory["exit_code"] or foreign or unexplained_memory > 128:
            raise RuntimeError("exclusive policy GPU unavailable; retained samples are incomplete")
    adapter, runtimes, rows = None, {}, []
    try:
        resources("admission")
        root = Path(__file__).resolve().parents[2]
        adapter = create_policy(dict(model_root=str(root), model_config=str(root / "configs/dreamwam_joint.yaml"),
            checkpoint=str(root / "checkpoints/dreamwam_joint.pt"), action_horizon=32, denoising_steps=10,
            rng_mode="fixed_per_predict", prompt_cache={"capacity": 8}, visual_cache=dict(refresh_every=1,
                conditioned_frame_reuse=True, graph_dispatch="all_transformers")), "cuda")
        report["dense_adapter_fingerprint"] = adapter.describe().fingerprint
        if report["dense_adapter_fingerprint"]["checkpoint_sha256"] != "6c087e5b9e201f19dbe92834b470a7ee18937a18ea9adcbb49886aa1e3ac4c61":
            raise AssertionError("checkpoint differs from frozen reference")
        policy = adapter.policy
        versions = {n: value._version for n, value in policy.model.named_parameters()}
        evaluation = dict(policy.evaluation)
        prompt = policy._prompt_cache_runtime
        runtimes["dense"] = policy._visual_cache_runtime
        runtimes["dense"].__exit__(None, None, None)
        runtimes.update({name: HybridVisualRuntime(policy.model, config) for name, config in configs.items()})
        controllers = {name: ObservationBudget(config) for name, config in budgets.items()}
        names, refs, decision_refs = list(runtimes), {}, {}
        def graph_identity(runtime):
            return {k: (v["captured"], v["capture_seconds"]) for k, v in runtime.graph_stats().items()}
        def replay(name, runtime, identity, sequence, phase, repeat):
            policy._hybrid_visual_runtime = runtime if name != "dense" else None
            policy._visual_cache_runtime = runtime if name == "dense" else None
            policy._chunk_budget_runtime = controllers.get(name)
            adapter.reset(dict(episode_id=str(identity), seed=identity[-1]))
            with runtime:
                for index, (entry, observation) in enumerate(sequence):
                    before = graph_identity(runtime) if phase == "timed" else None
                    torch.cuda.synchronize(); start = time.perf_counter()
                    prediction = adapter.predict(observation)
                    torch.cuda.synchronize(); elapsed = time.perf_counter() - start
                    stats = prediction.diagnostics
                    work = stats["visual_cache" if name == "dense" else "hybrid_visual"]
                    key = (name, entry["id"])
                    budget = stats.get("chunk_budget", {})
                    decisions = (budget.get("level_index"), budget.get("features"),
                        budget.get("history_length"), work["computed_video_token_layers"],
                        [s["effective_op"] for s in work.get("steps", [])])
                    if phase == "reference":
                        refs[key], decision_refs[key] = prediction.actions.copy(), decisions
                    elif not np.array_equal(prediction.actions, refs[key]) or decisions != decision_refs[key]:
                        raise AssertionError("own uncached eager action or causal decision mismatch: " + str(key))
                    if work["action_layer_updates"] != 300:
                        raise AssertionError("action budget changed")
                    if name != "dense" and runtime.state.output is not None:
                        raise AssertionError("visual cache leaked across chunks")
                    if phase == "timed":
                        if graph_identity(runtime) != before:
                            raise AssertionError("graph capture or eviction contaminated a timed request")
                        if stats["prompt_cache"]["last_hit"] != (index > 0):
                            raise AssertionError("unexpected prompt-cache boundary")
                        row = dict(variant=name, repeat=repeat, input_id=entry["id"], seconds=elapsed,
                            prompt_hit=index > 0, own_uncached_eager_bitwise=True, diagnostics=stats,
                            action_difference=diagnostics(prediction.actions, refs["dense", entry["id"]]))
                        rows.append(row); journal.write(json.dumps(row, allow_nan=False) + "\n"); journal.flush()
                    elif phase == "setup":
                        report["setup"].append(dict(variant=name, input_id=entry["id"], seconds=elapsed))
            policy._hybrid_visual_runtime = policy._visual_cache_runtime = None
            policy._chunk_budget_runtime = None
        policy.text_encoder = prompt.encoder
        for name in names:
            eager = ConditionedFrameCache(policy.model, refresh_every=1) if name == "dense" else HybridVisualRuntime(policy.model, replace(configs[name], backend="eager"))
            for identity, sequence in sequences.items(): replay(name, eager, identity, sequence, "reference", -1)
        policy.text_encoder = prompt
        np.savez_compressed(args.out_dir / "references.npz", **{name + "__" + key: value for (name, key), value in refs.items()})
        report["references_sha256"] = sha256(args.out_dir / "references.npz")
        for name in names:
            for identity, sequence in sequences.items(): replay(name, runtimes[name], identity, sequence, "setup", -1)
        with (args.out_dir / "requests.jsonl").open("w", buffering=1) as journal:
            for repeat in range(args.reps):
                for identity, sequence in sequences.items():
                    order = names[repeat % len(names):] + names[:repeat % len(names)]
                    for name in order:
                        resources(f"repeat={repeat} episode={identity} arm={name}")
                        replay(name, runtimes[name], identity, sequence, "timed", repeat)
                    print(json.dumps(dict(repeat=repeat, episode=identity, complete_timings=len(rows))), flush=True)
        resources("completed")
        if evaluation != policy.evaluation or versions != {n: p._version for n, p in policy.model.named_parameters()}:
            raise AssertionError("protocol or weights changed")
        for name in names:
            selected = [r for r in rows if r["variant"] == name]
            warm = [r["seconds"] for r in selected if r["prompt_hit"]]
            cold = [r["seconds"] for r in selected if not r["prompt_hit"]]
            report["summary"][name] = dict(calls=len(selected), warm_calls=len(warm), warm_mean_ms=float(np.mean(warm)*1000),
                warm_p50_ms=float(np.median(warm)*1000), warm_p95_ms=float(np.quantile(warm,.95)*1000),
                episode_first_mean_ms=float(np.mean(cold)*1000), all_mean_ms=float(np.mean([r["seconds"] for r in selected])*1000),
                levels=dict(Counter(r["diagnostics"].get("chunk_budget", {}).get("level", "dense") for r in selected)))
        for summary in report["summary"].values():
            summary["warm_ratio_vs_strong_dense"] = report["summary"]["dense"]["warm_mean_ms"] / summary["warm_mean_ms"]
        report.update(status="COMPLETE", exit_code=0, timed_requests=len(rows))
    except BaseException as exc:
        report.update(status="ERROR", exit_code=1, error=repr(exc), accepted_timings=len(rows))
        raise
    finally:
        for runtime in runtimes.values():
            runtime.__exit__(None, None, None); runtime.close_graphs()
        if adapter is not None:
            adapter.policy._hybrid_visual_runtime = adapter.policy._visual_cache_runtime = None
            adapter.policy._chunk_budget_runtime = None
            adapter.close()
        report["end_utc"] = datetime.now(timezone.utc).isoformat(); save()
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
