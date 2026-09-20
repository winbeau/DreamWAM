#!/usr/bin/env python3
"""Finite native-route screening with contemporaneous controls and raw actions.

Only the frozen, historically exposed development observations are used. Every
prediction is journaled, including eager/capture/prompt-miss setup. No simulator,
closed-loop success computation, automatic retry, or candidate expansion occurs.
"""

import argparse
from datetime import datetime, timezone
from dataclasses import replace
import json
import os
from pathlib import Path
import signal
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from benchmark_ffn_context_cache import command
from benchmark_hybrid_schedules import close_runtime, make_runtime, write_json, diagnostics
from dreamwam.config import load_release_config
from dreamwam.policy import build_policy
from dreamwam.sparse.hybrid.experiment import request_cells, summarize
from dreamwam.sparse.hybrid.native_experiment import frozen_candidates, native_candidates, planned_calls, uniform_feature_control
from dreamwam.sparse.hybrid.runtime import HybridVisualRuntime
from dreamwam.sparse.profile.admission import admit_profile
from dreamwam.sparse.profile.archive import sha256
from dreamwam.sparse.profile.study import error_metrics


def stamp():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("selectors", "layers", "pooling", "refresh", "structure", "frozen"), required=True)
    parser.add_argument("--design-key", help="committed plan section containing an explicit frozen candidate matrix")
    parser.add_argument("--plan", type=Path, default=Path("docs/implementation/dido-sparse-profile/experiment-plan.json"))
    parser.add_argument("--config", type=Path, default=Path("configs/dreamwam_joint.yaml"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--share-gpu5", action="store_true")
    parser.add_argument("--admission-wait-seconds", type=int, default=0,
                        help="bounded pre-load admission wait, 0..120 seconds; no prediction retries")
    args = parser.parse_args()
    if not 0 <= args.admission_wait_seconds <= 120:
        parser.error("admission wait must be within 0..120 seconds")
    plan = json.loads(args.plan.read_text())
    if bool(args.design_key) != (args.stage == "frozen"):
        parser.error("--design-key is required exactly for --stage frozen")
    design = plan[args.design_key or ("online_followup" if args.stage in ("refresh", "structure") else "online_screen")]
    inputs_path = Path(plan["development"]["profile_inputs"])
    if sha256(inputs_path) != plan["development"]["profile_inputs_sha256"]:
        parser.error("development input manifest changed")
    captured = json.loads(inputs_path.read_text())
    if captured.get("status") != "CAPTURED":
        parser.error("require captured development observations")
    observations = {}
    input_metadata = {}
    indexed = {item["id"]: item for item in captured["inputs"]}
    for name in design["input_ids"]:
        item = indexed[name]
        path = inputs_path.parent / item["path"]
        if sha256(path) != item["sha256"]:
            parser.error("captured observation bytes changed")
        with np.load(path, allow_pickle=False) as data:
            observations[name] = dict(images={key: data[key].copy() for key in ("agentview", "wrist")},
                state=data["state"].copy(), instruction=str(data["instruction"].item()))
        input_metadata[name] = item
    if not 2 <= len(observations) <= 9 or design["repeats"] != 2 or design["group_size"] != 2:
        parser.error("finite screens require 2..9 declared inputs, two repeats and two candidates/group")
    candidates = frozen_candidates(design) if args.stage == "frozen" else native_candidates(args.stage)
    configs = {config.policy_hash: config for _, config in candidates}
    labels = {config.policy_hash: label for label, config in candidates}
    controls = ("dense_strong", "uniform_features")
    cells = request_cells(list(configs), list(observations), repeats=design["repeats"],
                          group_size=design["group_size"], controls=controls, seed=42)
    budget = planned_calls(cells, len(observations))
    declared = dict(predict_call_cap=design["predict_call_cap"], candidates=len(design["candidates"])) if args.stage == "frozen" else design["stages"][args.stage]
    if budget["total"] != declared["predict_call_cap"] or len(configs) != declared["candidates"]:
        parser.error("declared candidate/call cap differs from the frozen executable design")
    source = command("git", "rev-parse", "HEAD")
    dirty = command("git", "status", "--porcelain")
    if source["exit_code"] or dirty["exit_code"] or dirty["stdout"]:
        parser.error("require a clean immutable source checkout")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible:
        parser.error("select one explicitly authorized GPU UUID")
    release = load_release_config(args.config)
    if release.evaluation["denoising_steps"] != 10 or release.evaluation["action_horizon"] != 32:
        parser.error("released denoising steps/action horizon changed")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    (args.out_dir / "actions").mkdir()
    report = dict(status="PREPARING", exit_code=None, start_utc=stamp(), argv=sys.argv, pid=os.getpid(),
        source_commit=source["stdout"].strip(), plan_sha256=sha256(args.plan), config_sha256=sha256(args.config),
        input_manifest_sha256=sha256(inputs_path), inputs=input_metadata,
        input_kind="self_captured_observations", split_role="development", stage=args.design_key or args.stage,
        eager_reference_diagnostics="trace" if design.get("trace_eager", False) else "counters",
        torch=torch.__version__, cuda=torch.version.cuda, python=sys.version, gpu_uuid=visible,
        shared_host=True, selected_gpu_sharing=args.share_gpu5, gpu_snapshots=[],
        checkpoint_sha256=None, evaluation=dict(release.evaluation),
        controls={"dense_strong": "conditioned-frame cache plus CUDA graph; all 10 native denoising steps",
                  "uniform_features": uniform_feature_control().describe()},
        labels=labels, candidates={name: config.describe() for name, config in configs.items()},
        cells=cells, budget=budget, started_calls=0, completed_calls=0, raw_bytes=0, graphs={}, sr=None,
        closed_loop_episode_attempts=0,
        limitations=["historically exposed development observations, not confirmation",
            "two timing repeats/input; action error cannot establish SR",
            "all online operations and request-local transfers are inside full predict_action timing",
            "raw-output archival and hashing happen after timing; raw observer call overhead is included",
            "anchor-native scores refresh only at Dense steps; Sparse recomputation and structure reuse are separate controls",
            "fusion weights are competing calibration settings, not an assumed improvement"])
    write = lambda: write_json(args.out_dir / "report.json", report)
    write()
    policy, runtimes = None, {}
    accepted = {}
    call_file = (args.out_dir / "calls.jsonl").open("x", buffering=1)
    timed_file = (args.out_dir / "requests.jsonl").open("x", buffering=1)
    current_group = -1

    def terminated(signum, frame):
        raise TimeoutError(f"finite job received signal {signum}")
    signal.signal(signal.SIGTERM, terminated)

    def predict(runtime, name, input_id, phase):
        if report["started_calls"] >= budget["total"]:
            raise RuntimeError("inclusive prediction cap reached")
        number = report["started_calls"]
        report["started_calls"] += 1
        row = dict(call_id=number, group=current_group, variant=name, input_id=input_id, phase=phase, start_utc=stamp())
        raw = {}
        try:
            with runtime:
                sample = policy.model.sample_action
                def observe(*positional, **keyword):
                    result = sample(*positional, **keyword)
                    raw["action"] = result
                    return result
                policy.model.sample_action = observe
                try:
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    action = policy.predict_action(**observations[input_id])
                    torch.cuda.synchronize()
                    seconds = time.perf_counter() - started
                finally:
                    policy.model.sample_action = sample
                counters = dict(runtime.last_stats)
            raw_action = raw["action"].detach().float().cpu().numpy().copy()
            if not np.isfinite(action).all() or not np.isfinite(raw_action).all():
                raise FloatingPointError("nonfinite action output")
            if counters["action_layer_updates"] != 300:
                raise AssertionError("native action update budget changed")
            path = args.out_dir / "actions" / f"call-{number:04d}.npz"
            np.savez_compressed(path, action=action, raw_action=raw_action)
            report["raw_bytes"] += action.nbytes + raw_action.nbytes
            if report["raw_bytes"] > design["raw_byte_cap"]:
                raise RuntimeError("raw action archive cap exceeded")
            row.update(status="COMPLETE", seconds=seconds, counters=counters,
                prompt_hit=policy._prompt_cache_runtime.last_hit, actions_path=str(path.relative_to(args.out_dir)),
                actions_sha256=sha256(path))
            report["completed_calls"] += 1
            return dict(action=action, raw_action=raw_action, record=row)
        except BaseException as exc:
            row.update(status="ERROR", error=repr(exc))
            raise
        finally:
            row["end_utc"] = stamp()
            call_file.write(json.dumps(row, allow_nan=False) + "\n")
            write()

    def same(actual, expected):
        if not all(np.array_equal(actual[key], expected[key]) for key in ("action", "raw_action")):
            raise AssertionError("executable or pre-binarization action differs from own eager reference")

    def runtime_for(name, backend):
        if name == "uniform_features":
            config = uniform_feature_control(backend)
            if backend == "eager" and design.get("trace_eager", False):
                config = replace(config, diagnostics="trace")
            return HybridVisualRuntime(policy.model, config)
        if name in configs and backend == "eager" and design.get("trace_eager", False):
            return HybridVisualRuntime(policy.model, replace(configs[name], backend="eager", diagnostics="trace"))
        return make_runtime(policy.model, name, configs, backend)

    def snapshot(phase):
        report["gpu_snapshots"].append(dict(timestamp_utc=stamp(), group=current_group, phase=phase,
            inventory=command("nvidia-smi", "-i", visible,
                "--query-gpu=uuid,memory.used,memory.free,utilization.gpu,clocks.sm,power.draw", "--format=csv,noheader"),
            processes=command("nvidia-smi", "-i", visible,
                "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader")))

    try:
        report["checkpoint_sha256"] = sha256(release.paths.checkpoint)
        if report["checkpoint_sha256"] != plan["checkpoint_sha256"]:
            raise ValueError("checkpoint differs from frozen profile/diagnostic cohort")
        # Admission is deliberately after all large-file hashing and immediately
        # before model initialization. Failure preserves its full inventory.
        deadline = time.monotonic() + args.admission_wait_seconds
        report["admission_observations"] = []
        while True:
            try:
                report["admission"] = admit_profile(visible, share_gpu5=args.share_gpu5)
                break
            except RuntimeError as exc:
                report["admission_observations"].append(dict(timestamp_utc=stamp(), error=str(exc)))
                write()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise
                print(json.dumps(dict(stage=args.stage, status="WAITING_FOR_ADMISSION",
                                      remaining_seconds=round(remaining, 1), model_loaded=False)), flush=True)
                time.sleep(min(5., remaining))
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("exactly one visible CUDA device is required")
        report["status"] = "RUNNING"
        write()
        started = time.perf_counter()
        policy = build_policy(release, device="cuda", prompt_cache={"capacity": 8})
        report["model_load_seconds"] = time.perf_counter() - started
        versions = {name: parameter._version for name, parameter in policy.model.named_parameters()}
        torch.cuda.reset_peak_memory_stats()
        for current_group in sorted({cell["group"] for cell in cells}):
            snapshot("group_start")
            group_cells = [cell for cell in cells if cell["group"] == current_group]
            variants = sorted({cell["variant"] for cell in group_cells})
            refs = {}
            for name in variants:
                eager = runtime_for(name, "eager")
                try:
                    for input_id in observations:
                        refs[name, input_id] = predict(eager, name, input_id, "eager_reference")
                finally:
                    close_runtime(eager)
                runtimes[name] = runtime_for(name, "cuda_graph")
            for name in variants:
                if labels.get(name) == "uniform":
                    for input_id in observations:
                        same(refs[name, input_id], refs["uniform_features", input_id])
            for name, runtime in runtimes.items():
                policy._prompt_cache_runtime.clear()
                for input_id in observations:
                    actual = predict(runtime, name, input_id, "graph_capture_or_changed_input")
                    same(actual, refs[name, input_id])
                policy._prompt_cache_runtime.clear()
                first = next(iter(observations))
                actual = predict(runtime, name, first, "warm_graph_cold_prompt")
                same(actual, refs[name, first])
                if actual["record"]["prompt_hit"]:
                    raise AssertionError("cold-prompt check unexpectedly hit the instruction cache")
            for input_id in observations:
                same(predict(runtimes["dense_strong"], "dense_strong", input_id, "restore_prompts"),
                     refs["dense_strong", input_id])
            for cell in group_cells:
                name, input_id = cell["variant"], cell["input_id"]
                actual = predict(runtimes[name], name, input_id, "timed")
                same(actual, refs[name, input_id])
                if not actual["record"]["prompt_hit"]:
                    raise AssertionError("warm timing had an instruction-cache miss")
                row = dict(**cell, attempt_id=0, own_eager_parity=True,
                    **{key: actual["record"][key] for key in ("call_id", "seconds", "counters", "prompt_hit", "actions_path", "actions_sha256")},
                    action_diagnostics=diagnostics(actual["action"], refs["dense_strong", input_id]["action"]),
                    executed_prefix=diagnostics(actual["action"][:10], refs["dense_strong", input_id]["action"][:10]),
                    raw_action=error_metrics(actual["raw_action"], refs["dense_strong", input_id]["raw_action"]),
                    raw_prefix=error_metrics(actual["raw_action"][:, :10], refs["dense_strong", input_id]["raw_action"][:, :10]))
                timed_file.write(json.dumps(row, allow_nan=False) + "\n")
                accepted[cell["request_id"]] = row
            for name, runtime in runtimes.items():
                if hasattr(runtime, "graph_stats"):
                    report["graphs"][f"{current_group}/{name}"] = runtime.graph_stats()
                close_runtime(runtime)
            runtimes.clear()
            snapshot("group_end")
            print(json.dumps(dict(stage=args.stage, group=current_group, calls=report["completed_calls"],
                                  call_cap=budget["total"], timed=len(accepted))), flush=True)
        if report["completed_calls"] != budget["total"]:
            raise AssertionError("completed prediction count differs from the declared design")
        if versions != {name: parameter._version for name, parameter in policy.model.named_parameters()}:
            raise AssertionError("checkpoint parameters changed")
        report.update(status="COMPLETE", exit_code=0)
    except BaseException as exc:
        report.update(status="ERROR", error=repr(exc), exit_code=1)
        raise
    finally:
        for name, runtime in runtimes.items():
            if hasattr(runtime, "graph_stats"):
                report["graphs"][f"{current_group}/{name}"] = runtime.graph_stats()
            close_runtime(runtime)
        if policy is not None:
            report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
            policy.close()
        snapshot("final")
        report["end_utc"] = stamp()
        call_file.close()
        timed_file.close()
        report["artifacts"] = {name: sha256(args.out_dir / name) for name in ("calls.jsonl", "requests.jsonl")}
        write()
        summary = summarize(cells, accepted, list(configs))
        summary.update(labels=labels, report_sha256=sha256(args.out_dir / "report.json"),
                       actual_completed_calls=report["completed_calls"], declared_call_cap=budget["total"])
        write_json(args.out_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
