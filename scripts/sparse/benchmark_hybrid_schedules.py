#!/usr/bin/env python3
"""Finite, resumable hybrid schedule screening on hash-verified real observations.

Each small candidate group has contemporaneous Dense/legacy/fresh controls.
All variants run complete predict_action calls; reference/first-miss/capture
calls are separate from the measured journal. This is NOT a closed-loop SR run.
"""

import argparse
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from benchmark_ffn_context_cache import command, sha256
from summarize_hybrid_schedules import report
from dreamwam.config import load_release_config
from dreamwam.policy import build_policy
from dreamwam.sparse.action_guided_visual_token_cache import ActionGuidedVisualTokenCache
from dreamwam.sparse.conditioned_frame_cache import ConditionedFrameCache, GraphedConditionedFrameCache
from dreamwam.sparse.fresh_visual_tokens import FreshVisualTokenSparsity
from dreamwam.sparse.hybrid.experiment import read_journal, request_cells
from dreamwam.sparse.hybrid.runtime import HybridVisualRuntime
from dreamwam.sparse.hybrid.schedule import stable_hash
from dreamwam.sparse.hybrid.search import read_candidates
from dreamwam.sparse.visual_cache_graphs import GraphedVisualTokenCache


def stamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def diagnostics(actual, dense):
    difference = actual.astype(np.float64) - dense.astype(np.float64)
    a, b = actual.astype(np.float64).ravel(), dense.astype(np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return dict(relative_l2=float(np.linalg.norm(difference) / max(nb, 1e-12)),
                cosine=float(np.dot(a, b) / (na * nb)) if na * nb else float(na == nb),
                max_abs=float(np.abs(difference).max()),
                component_mae=np.abs(difference).mean(axis=0).tolist(),
                gripper_sign_disagreements=int(np.count_nonzero(np.sign(actual[:, -1]) != np.sign(dense[:, -1]))))


def make_runtime(model, name, configs, backend):
    graph = backend == "cuda_graph"
    if name in configs:
        return HybridVisualRuntime(model, replace(configs[name], backend=backend))
    if name == "dense_strong":
        return (ConditionedFrameCache(model, refresh_every=1) if backend == "eager" else
                GraphedConditionedFrameCache(model, refresh_every=1, graph_enabled=graph))
    if name == "legacy":
        return (ActionGuidedVisualTokenCache(model, keep_ratio=0.1, refresh_every=5, guidance_weight=1)
                if backend == "eager" else GraphedVisualTokenCache(
                    model, keep_ratio=0.1, refresh_every=5, guidance_weight=1,
                    graph_enabled=graph, graph_partial=True))
    if name == "fresh":
        return FreshVisualTokenSparsity(model, keep_ratio=0.1, selection="action",
                                       graph_enabled=graph, buffered=backend != "eager")
    raise ValueError(f"unknown control {name}")


def close_runtime(runtime):
    if hasattr(runtime, "close_graphs"):
        runtime.close_graphs()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--schedules", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--config", default="configs/dreamwam_joint.yaml")
    p.add_argument("--split-label", choices=("debug", "development", "confirmation"), required=True)
    p.add_argument("--reps", type=int, default=6)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--group-size", type=int, default=2)
    p.add_argument("--seed", type=int, default=42, help="experiment ordering only; model RNG unchanged")
    p.add_argument("--controls", default="dense_strong,legacy,fresh")
    p.add_argument("--max-candidates", type=int)
    p.add_argument("--max-new-requests", type=int)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--allow-shared-gpu", action="store_true")
    p.add_argument("--trace", action="store_true", help="separate untimed hybrid phase traces")
    args = p.parse_args()
    if args.reps < 1 or args.warmup < 1 or args.group_size < 1:
        p.error("reps, warmup and group-size must be positive")
    if any(value is not None and value < 1 for value in (args.max_candidates, args.max_new_requests)):
        p.error("finite request/candidate limits must be positive")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        p.error("select exactly one explicitly authorized GPU using CUDA_VISIBLE_DEVICES")
    inventory = command("nvidia-smi", "-i", visible, "--query-gpu=uuid,name,memory.used,utilization.gpu", "--format=csv,noheader")
    processes = command("nvidia-smi", "-i", visible, "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader")
    if inventory["exit_code"] or processes["exit_code"]:
        p.error("GPU inventory unavailable")
    foreign = [line for line in processes["stdout"].splitlines() if not line.startswith(str(os.getpid()) + ",")]
    if foreign and not args.allow_shared_gpu:
        p.error("GPU has other processes; explicit --allow-shared-gpu required")
    candidates = read_candidates(args.schedules)
    selected = candidates[:args.max_candidates] if args.max_candidates else candidates
    configs = dict(selected)
    backends = {config.backend for config in configs.values()}
    if len(backends) != 1 or any(config.diagnostics != "counters" for config in configs.values()):
        p.error("one backend per search manifest; timed candidates must use counters diagnostics")
    backend = next(iter(backends))
    controls = tuple(args.controls.split(","))
    if any(name not in ("dense_strong", "legacy", "fresh") for name in controls):
        p.error("unsupported control")
    captured = json.loads(args.inputs.read_text())
    if captured.get("status") != "CAPTURED" or not captured.get("inputs"):
        p.error("require captured real inputs with hashes")
    if args.split_label == "confirmation" and (captured.get("split_role") != "confirmation"
                                               or not captured.get("episode_split_sha256")):
        p.error("confirmation requires an explicitly frozen episode split in the input manifest")
    observations = {}
    for item in captured["inputs"]:
        name = item["id"]
        if not isinstance(name, str) or not name or name in observations:
            p.error("input ids must be nonempty unique strings")
        path = args.inputs.parent / item["path"]
        if sha256(path) != item["sha256"]:
            p.error("captured input hash mismatch")
        with np.load(path, allow_pickle=False) as data:
            observations[name] = dict(images={k: data[k].copy() for k in ("agentview", "wrist")},
                                      state=data["state"].copy(), instruction=str(data["instruction"].item()))
    release = load_release_config(args.config)
    if any(config.schedule.num_steps != release.evaluation["denoising_steps"] for config in configs.values()):
        p.error("candidate schedule does not match the unchanged release denoising_steps")
    cells = request_cells(list(configs), list(observations), repeats=args.reps,
                          group_size=args.group_size, controls=controls, seed=args.seed)
    git = command("git", "rev-parse", "HEAD")
    dirty = command("git", "status", "--porcelain")
    if git["exit_code"] or dirty["exit_code"] or dirty["stdout"]:
        p.error("benchmark requires a clean committed checkout")
    identity_data = dict(git=git["stdout"], checkpoint_sha256=sha256(release.paths.checkpoint),
        config_sha256=sha256(args.config), schedules_sha256=sha256(args.schedules),
        inputs_sha256=sha256(args.inputs), selected_candidates=list(configs),
        gpu=inventory["stdout"].split(",")[0], torch=torch.__version__, cuda=torch.version.cuda,
        split_label=args.split_label, backend=backend, reps=args.reps, warmup=args.warmup,
        group_size=args.group_size, seed=args.seed, controls=controls, trace=args.trace,
        allow_shared_gpu=args.allow_shared_gpu)
    identity = stable_hash(identity_data)
    if args.resume and not args.out_dir.is_dir():
        p.error("resume requires an existing output directory")
    if not args.resume:
        args.out_dir.mkdir(parents=True, exist_ok=False)
    lock = (args.out_dir / ".lock").open("a")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest_path = args.out_dir / "manifest.json"
    if args.resume:
        manifest = json.loads(manifest_path.read_text())
        if manifest["identity"] != identity:
            p.error("resume source/input/hardware/config identity mismatch")
    else:
        manifest = dict(schema_version=1, identity=identity, identity_data=identity_data,
            created_utc=stamp(), status="PREPARING", cells=cells,
            candidates={key: value.describe() for key, value in configs.items()},
            all_manifest_candidates=len(candidates), searched_candidates=len(configs),
            attempts=[], setup=[], references={}, graphs={}, sr=None,
            limitations=["open-loop input replay; action error is not SR",
                "candidate groups have contemporaneous controls; do not pool hardware cohorts",
                "cyclic ordering balances counts, not necessarily every order position",
                "first prompt misses and graph capture are excluded from warm request samples"])
    journal_path = args.out_dir / "requests.jsonl"
    completed = read_journal(journal_path, cells)
    # Resume verifies every accepted action artifact instead of silently accepting missing data.
    for row in completed.values():
        if sha256(args.out_dir / row["actions_path"]) != row["actions_sha256"]:
            raise ValueError("accepted action artifact changed")
    manifest["attempts"].append(dict(start_utc=stamp(), pid=os.getpid(), argv=sys.argv,
                                     gpu_before=inventory, processes_before=processes))
    manifest["status"] = "RUNNING"
    write_json(manifest_path, manifest)
    policy, runtimes = None, {}
    new_requests = 0
    try:
        if len(completed) == len(cells):
            manifest["status"] = "COMPLETE"
            return
        started = time.perf_counter()
        policy = build_policy(release, device="cuda", prompt_cache={"capacity": max(8, len(observations))})
        manifest["attempts"][-1]["model_load_seconds"] = time.perf_counter() - started
        manifest["evaluation"] = dict(policy.evaluation)
        weights = {name: parameter._version for name, parameter in policy.model.named_parameters()}
        torch.cuda.reset_peak_memory_stats()
        (args.out_dir / "actions").mkdir(exist_ok=True)
        (args.out_dir / "references").mkdir(exist_ok=True)
        groups = defaultdict(list)
        for cell in cells:
            groups[cell["group"]].append(cell)

        def predict(runtime, input_id):
            with runtime:
                torch.cuda.synchronize()
                started = time.perf_counter()
                action = policy.predict_action(**observations[input_id])
                torch.cuda.synchronize()
                seconds = time.perf_counter() - started
                counters = dict(runtime.last_stats)
            if not np.isfinite(action).all():
                raise FloatingPointError("non-finite action")
            if counters["action_layer_updates"] != release.evaluation["denoising_steps"] * policy.model.mot.num_layers:
                raise AssertionError("action layer update budget changed")
            return seconds, action, counters

        with journal_path.open("a", buffering=1) as journal:
            for group, group_cells in groups.items():
                if all(cell["request_id"] in completed for cell in group_cells):
                    continue
                variants = sorted({cell["variant"] for cell in group_cells})
                refs = {}
                for variant in variants:
                    eager = make_runtime(policy.model, variant, configs, "eager")
                    try:
                        for input_id in observations:
                            _, action, _ = predict(eager, input_id)
                            refs[variant, input_id] = action
                            ref_key = stable_hash([variant, input_id])
                            path = args.out_dir / "references" / (ref_key + ".npy")
                            if path.exists():
                                if not np.array_equal(np.load(path, allow_pickle=False), action):
                                    raise AssertionError("own eager reference changed across resume/groups")
                            else:
                                np.save(path, action, allow_pickle=False)
                            manifest["references"][ref_key] = dict(variant=variant, input_id=input_id,
                                path=str(path.relative_to(args.out_dir)), sha256=sha256(path))
                    finally:
                        close_runtime(eager)
                    runtimes[variant] = make_runtime(policy.model, variant, configs, backend)
                for variant, runtime in runtimes.items():
                    for input_id in observations:
                        for warmup in range(args.warmup):
                            seconds, actions, counters = predict(runtime, input_id)
                            if not np.array_equal(actions, refs[variant, input_id]):
                                raise AssertionError("dispatch/eager parity failed during setup")
                            manifest["setup"].append(dict(group=group, variant=variant, input_id=input_id,
                                phase="warmup", warmup=warmup, seconds=seconds,
                                prompt_hit=policy._prompt_cache_runtime.last_hit))
                    first_input = next(iter(observations))
                    policy._prompt_cache_runtime.clear()
                    seconds, actions, _ = predict(runtime, first_input)
                    if policy._prompt_cache_runtime.last_hit or not np.array_equal(actions, refs[variant, first_input]):
                        raise AssertionError("first instruction miss validation failed")
                    manifest["setup"].append(dict(group=group, variant=variant, input_id=first_input,
                                                  phase="prompt_miss_warm_graph", seconds=seconds))
                # Restore instruction hits without changing observations or model state.
                for input_id in observations:
                    predict(runtimes["dense_strong"], input_id)
                write_json(manifest_path, manifest)
                for cell in group_cells:
                    if cell["request_id"] in completed:
                        continue
                    variant, input_id = cell["variant"], cell["input_id"]
                    seconds, actions, counters = predict(runtimes[variant], input_id)
                    if not np.array_equal(actions, refs[variant, input_id]):
                        raise AssertionError("timed output differs from own eager reference")
                    if variant in configs and counters["plan_hash"] != variant:
                        raise AssertionError("executed plan differs from candidate identity")
                    if variant in configs and counters["dense_steps"] == release.evaluation["denoising_steps"]:
                        if not np.array_equal(actions, refs["dense_strong", input_id]):
                            raise AssertionError("all-dense hybrid control differs from stronger Dense")
                    action_path = args.out_dir / "actions" / (cell["request_id"] + ".npy")
                    np.save(action_path, actions, allow_pickle=False)
                    row = dict(**cell, attempt_id=len(manifest["attempts"]) - 1,
                        seconds=seconds, counters=counters, own_eager_parity=True,
                        prompt_hit=policy._prompt_cache_runtime.last_hit,
                        actions_path=str(action_path.relative_to(args.out_dir)), actions_sha256=sha256(action_path),
                        action_diagnostics=diagnostics(actions, refs["dense_strong", input_id]))
                    journal.write(json.dumps(row, allow_nan=False) + "\n")
                    journal.flush()
                    os.fsync(journal.fileno())
                    completed[cell["request_id"]] = row
                    new_requests += 1
                    if args.max_new_requests and new_requests >= args.max_new_requests:
                        manifest["status"] = "COMPLETE" if len(completed) == len(cells) else "PARTIAL"
                        return
                if args.trace:
                    (args.out_dir / "traces").mkdir(exist_ok=True)
                    for variant in variants:
                        if variant not in configs:
                            continue
                        traced = HybridVisualRuntime(policy.model, replace(configs[variant], diagnostics="trace"))
                        try:
                            for input_id in observations:
                                predict(traced, input_id)  # Capture outside the recorded diagnostic pass.
                                _, actions, stats = predict(traced, input_id)
                                if not np.array_equal(actions, refs[variant, input_id]):
                                    raise AssertionError("trace changed actions")
                                write_json(args.out_dir / "traces" / (stable_hash([variant, input_id]) + ".json"),
                                    dict(candidate_id=variant, input_id=input_id, stats=stats,
                                         semantics="instrumented CPU and CUDA stream spans; not kernel self-time"))
                        finally:
                            traced.close_graphs()
                for name, runtime in runtimes.items():
                    if hasattr(runtime, "graph_stats"):
                        manifest["graphs"][f"{group}/{name}"] = runtime.graph_stats()
                    close_runtime(runtime)
                runtimes.clear()
                write_json(manifest_path, manifest)
                print(json.dumps(dict(group=group, completed=len(completed), planned=len(cells))), flush=True)
        if weights != {name: parameter._version for name, parameter in policy.model.named_parameters()}:
            raise AssertionError("model parameters changed")
        manifest["status"] = "COMPLETE"
    except BaseException as exc:
        manifest.update(status="ERROR", error=repr(exc))
        raise
    finally:
        for name, runtime in runtimes.items():
            if hasattr(runtime, "graph_stats"):
                manifest["graphs"][f"{group}/{name}"] = runtime.graph_stats()
            close_runtime(runtime)
        if policy is not None:
            manifest["attempts"][-1]["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            policy.close()
        manifest["attempts"][-1].update(end_utc=stamp(), status=manifest["status"],
            completed_requests=len(completed),
            gpu_after=command("nvidia-smi", "-i", visible, "--query-gpu=uuid,memory.used,utilization.gpu", "--format=csv,noheader"),
            processes_after=command("nvidia-smi", "-i", visible, "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader"))
        write_json(manifest_path, manifest)
        report(args.out_dir)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    main()
