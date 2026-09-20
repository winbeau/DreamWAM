#!/usr/bin/env python3
"""Independently replay bounded native-screen archives and export per-input metrics."""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from dreamwam.sparse.hybrid.config import HybridConfig
from dreamwam.sparse.hybrid.experiment import distribution, read_journal
from dreamwam.sparse.hybrid.native_experiment import planned_calls, uniform_feature_control
from dreamwam.sparse.hybrid.schedule import stable_hash
from dreamwam.sparse.profile.archive import sha256


def l2(actual, reference):
    actual, reference = actual.astype(np.float64), reference.astype(np.float64)
    return float(np.linalg.norm(actual - reference) / max(np.linalg.norm(reference), 1e-12))


def raw_output(root, record):
    path = root / record["actions_path"]
    if not path.resolve().is_relative_to((root / "actions").resolve()):
        raise ValueError("archive escapes its declared action directory")
    if sha256(path) != record["actions_sha256"]:
        raise ValueError("raw action archive hash mismatch")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"action", "raw_action"}:
            raise ValueError("raw action archive fields changed")
        result = {name: archive[name].copy() for name in archive.files}
    for name, shape in (("action", (32, 7)), ("raw_action", (1, 32, 7))):
        value = result[name]
        if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError("native action shape/dtype/finite contract changed")
    return result


def audit_native_counters(config, counters):
    """Reject feature-cache accounting passed off as fresh structure execution."""
    q_count, kv_count = config.budgets(294, 98)
    steps = counters["steps"]
    if len(steps) != 10 or counters["action_layer_updates"] != 300:
        raise ValueError("native step/action budget changed")
    computed, read, anchor = 0, 0, 0
    for index, (op, step) in enumerate(zip(config.schedule.operations, steps)):
        if op == "sparse" and q_count == kv_count == 294:
            op = "dense"
        if op == "dense":
            q, kv, anchor = 294, 294, index
        else:
            q = kv_count if config.reuse_mode == "structure" else q_count if op == "sparse" else 0
            kv = kv_count
        if (step["effective_op"] != op or step["q_rows"] != q or step["kv_rows"] != kv or
            step["native_score_age"] != index - anchor or
            step["reuse_mode"] != config.reuse_mode or
            step["reused_visual_features"] != (op == "reuse" and config.reuse_mode == "features")):
            raise ValueError("native counters differ from the frozen schedule/reuse design")
        computed += q * 30
        read += kv * 30
    if counters["computed_video_token_layers"] != computed or counters["read_video_token_layers"] != read:
        raise ValueError("native total token/layer budget changed")


def audit_route_trace(config, counters):
    """Verify archived hard-route membership, including newly read Sparse rows."""
    if config.native_routing and config.native_routing.packing != "hard":
        raise ValueError("this route trace audit covers hard selection, not pooled groups")
    q_count, read_count = config.budgets(294, 98)
    previous = None
    for step in counters["steps"]:
        route = np.asarray(step["route"])
        if (route.dtype.kind not in "iu" or route.ndim not in (1, 2) or
            route.shape[-1] != read_count or (route < 0).any() or (route >= 294).any() or
            stable_hash(step["route"]) != step["route_hash"]):
            raise ValueError("invalid archived route shape/range/hash")
        layerwise = config.native_routing is not None and config.native_routing.layerwise
        if route.shape != ((30, read_count) if layerwise else (read_count,)):
            raise ValueError("archived route depth layout changed")
        for layer in route.reshape(-1, read_count):
            if len(set(layer.tolist())) != read_count or any(not np.any(layer // 98 == frame) for frame in range(3)):
                raise ValueError("route duplicates rows or omits a frame")
        op = step["effective_op"]
        if op == "sparse" and config.reuse_mode == "features":
            query = np.asarray(step["query"])
            if route.ndim != 1 or query.shape != (q_count,) or query.dtype.kind not in "iu" or len(set(query.tolist())) != q_count:
                raise ValueError("Sparse query budget or route layout changed")
            new, old, updated = set(route.tolist()), set(previous.tolist()), set(query.tolist())
            if not new - old <= updated <= new:
                raise ValueError("newly read row was not freshly recomputed, or query is outside read set")
        elif op == "reuse" and previous is not None and not np.array_equal(previous, route):
            raise ValueError("Reuse step unexpectedly changed its structure")
        # At a shared Dense anchor, the historical query field denotes the
        # route seed; q_rows=294 records the actual full Dense computation.
        previous = route
    return len(counters["steps"])


def audit(root):
    root = Path(root)
    report = json.loads((root / "report.json").read_text())
    if report["status"] != "COMPLETE" or report["exit_code"] != 0:
        raise ValueError("audit completed finite screens; preserve unstarted/error attempts separately")
    if report["split_role"] != "development" or report["input_kind"] != "self_captured_observations":
        raise ValueError("screen provenance differs from the declared development cohort")
    if report["closed_loop_episode_attempts"] != 0 or report["sr"] is not None:
        raise ValueError("open-loop screen cannot carry a closed-loop SR")
    for name, expected in report["artifacts"].items():
        if sha256(root / name) != expected:
            raise ValueError("screen journal hash mismatch")
    calls = [json.loads(line) for line in (root / "calls.jsonl").read_text().splitlines()]
    budget = planned_calls(report["cells"], len(report["inputs"]))
    if (budget != report["budget"] or len(calls) != budget["total"] or
        report["started_calls"] != len(calls) or report["completed_calls"] != len(calls) or
        [row["call_id"] for row in calls] != list(range(len(calls)))):
        raise ValueError("inclusive prediction coverage differs from the frozen design")
    phases = dict(eager_reference="eager", graph_capture_or_changed_input="capture",
                  warm_graph_cold_prompt="prompt_miss", restore_prompts="restore_prompts", timed="timed")
    for phase, field in phases.items():
        if sum(row["phase"] == phase for row in calls) != budget[field]:
            raise ValueError("setup or timed call coverage changed")
    arrays, refs, dense = {}, {}, {}
    raw_bytes = 0
    for row in calls:
        if row["status"] != "COMPLETE" or (row["variant"] != "dense_strong" and row["counters"].get("status") != "COMPLETE"):
            raise ValueError("completed screen contains an unsuccessful call")
        arrays[row["call_id"]] = raw_output(root, row)
        raw_bytes += sum(array.nbytes for array in arrays[row["call_id"]].values())
        identity = (row["group"], row["variant"], row["input_id"])
        if row["phase"] == "eager_reference":
            if identity in refs:
                raise ValueError("duplicate eager reference")
            refs[identity] = arrays[row["call_id"]]
    if raw_bytes != report["raw_bytes"] or len(list((root / "actions").glob("*.npz"))) != len(calls):
        raise ValueError("raw bytes or archive-file coverage changed")
    traced_calls, traced_steps = 0, 0
    for row in calls:
        actual = arrays[row["call_id"]]
        reference = refs[row["group"], row["variant"], row["input_id"]]
        if not all(actual[name].tobytes() == reference[name].tobytes() for name in actual):
            raise ValueError("archived graph/setup output differs from its own eager reference")
        if row["variant"] == "dense_strong":
            previous = dense.setdefault(row["input_id"], actual)
            if not all(actual[name].tobytes() == previous[name].tobytes() for name in actual):
                raise ValueError("contemporaneous Dense reference changed across groups")
        if report["labels"].get(row["variant"]) == "uniform":
            control = refs[row["group"], "uniform_features", row["input_id"]]
            if not all(actual[name].tobytes() == control[name].tobytes() for name in actual):
                raise ValueError("native uniform differs from the inherited uniform control")
        if row["counters"]["action_layer_updates"] != 300:
            raise ValueError("action denoising/layer budget changed")
        if row["phase"] == "eager_reference" and report.get("eager_reference_diagnostics") == "trace" and row["variant"] != "dense_strong":
            trace_config = (uniform_feature_control() if row["variant"] == "uniform_features" else
                            HybridConfig.from_mapping(report["candidates"][row["variant"]]))
            traced_steps += audit_route_trace(trace_config, row["counters"])
            traced_calls += 1
    timed = read_journal(root / "requests.jsonl", report["cells"])
    if len(timed) != budget["timed"]:
        raise ValueError("timed journal is incomplete")
    metrics, checked_native = [], 0
    for row in timed.values():
        call = calls[row["call_id"]]
        if call["phase"] != "timed" or any(row[key] != call[key] for key in
                ("group", "variant", "input_id", "seconds", "counters", "actions_path", "actions_sha256")):
            raise ValueError("timing row differs from its archived call")
        actual, reference = arrays[row["call_id"]], dense[row["input_id"]]
        replay = dict(action_l2=l2(actual["action"], reference["action"]),
            prefix_l2=l2(actual["action"][:10], reference["action"][:10]),
            raw_l2=l2(actual["raw_action"], reference["raw_action"]),
            raw_prefix_l2=l2(actual["raw_action"][:, :10], reference["raw_action"][:, :10]),
            prefix_gripper_disagreements=int(np.count_nonzero(np.sign(actual["action"][:10, -1]) != np.sign(reference["action"][:10, -1]))))
        for field, section in (("action_l2", "action_diagnostics"), ("prefix_l2", "executed_prefix"),
                               ("raw_l2", "raw_action"), ("raw_prefix_l2", "raw_prefix")):
            if not np.isclose(replay[field], row[section]["relative_l2"], rtol=1e-12, atol=1e-12):
                raise ValueError("archived action metric replay mismatch")
        if replay["prefix_gripper_disagreements"] != row["executed_prefix"]["gripper_sign_disagreements"]:
            raise ValueError("executed-prefix gripper metric changed")
        config = report["candidates"].get(row["variant"])
        if config is not None:
            config = HybridConfig.from_mapping(config)
            if config.policy_hash != row["variant"]:
                raise ValueError("candidate identity changed")
            audit_native_counters(config, row["counters"])
            checked_native += 1
        metrics.append(dict(stage=report["stage"], group=row["group"], repeat=row["repeat"],
            input_id=row["input_id"], variant=row["variant"], label=report["labels"].get(row["variant"], row["variant"]),
            seconds=row["seconds"], **replay))
    groups = {}
    for row in metrics:
        groups.setdefault(row["variant"], []).append(row)
    lookup = {(row["group"], row["repeat"], row["input_id"], row["variant"]): row for row in metrics}
    summaries = []
    for name in report["candidates"]:
        rows = groups[name]
        controls = {control: [lookup[row["group"], row["repeat"], row["input_id"], control] for row in rows]
                    for control in ("dense_strong", "uniform_features")}
        unique = {row["input_id"]: row for row in rows}
        summaries.append(dict(variant=name, label=report["labels"][name],
            seconds=distribution([row["seconds"] for row in rows]),
            paired_speedup={control: sum(row["seconds"] for row in matches) / sum(row["seconds"] for row in rows)
                            for control, matches in controls.items()},
            inputs={key: {metric: row[metric] for metric in replay} for key, row in unique.items()},
            mean_raw_prefix_l2=statistics.fmean(row["raw_prefix_l2"] for row in unique.values()),
            worst_raw_prefix_l2=max(row["raw_prefix_l2"] for row in unique.values()),
            fallback_events=sum(row["counters"].get("selector_fallback_events", 0) for row in timed.values() if row["variant"] == name)))
    return dict(status="VERIFIED", root=str(root), report_sha256=sha256(root / "report.json"),
        source_commit=report["source_commit"], checkpoint_sha256=report["checkpoint_sha256"],
        input_manifest_sha256=report["input_manifest_sha256"], stage=report["stage"], gpu_uuid=report["gpu_uuid"],
        verified_calls=len(calls), verified_timed=len(timed), verified_native_counter_rows=checked_native,
        verified_trace_calls=traced_calls, verified_trace_steps=traced_steps,
        candidates=summaries, metrics=metrics, sr=None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    start = datetime.now(timezone.utc).isoformat()
    cohorts = [audit(path) for path in args.screen]
    rows = [row for cohort in cohorts for row in cohort.pop("metrics")]
    table = args.out_dir / "per-input-timings.csv"
    with table.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = dict(status="VERIFIED", start_utc=start, end_utc=datetime.now(timezone.utc).isoformat(),
        exit_code=0, argv=sys.argv, source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        cohorts=cohorts, artifacts={table.name: sha256(table)},
        limitations=["two exposed development observations per stage; timing repetitions are not independent task samples",
            "each candidate's frozen schedule/reuse mode is audited; no SR claim from action-vector replay",
            "all phase/raw files verified; reported speedups refer to warm complete predictions with prompt hits",
            "full cold/graph/prompt-miss calls remain separately recorded in source calls.jsonl"])
    (args.out_dir / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(status=result["status"], cohorts=len(cohorts), verified_calls=sum(cohort["verified_calls"] for cohort in cohorts)), indent=2))


if __name__ == "__main__":
    main()
