#!/usr/bin/env python3
"""Verify every frozen diagnostic cell and reproduce metrics from saved arrays."""

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from dreamwam.sparse.profile.analysis import index_capture, load_record, stamp, write_json
from dreamwam.sparse.profile.archive import sha256
from dreamwam.sparse.profile.study import output_diagnostics


def audit(root, out_dir):
    out_dir.mkdir(parents=True, exist_ok=False)
    evidence = dict(status="RUNNING", start_utc=stamp(), argv=sys.argv,
        source_commit=subprocess.check_output(["git", "-C", str(Path(__file__).resolve().parents[2]),
            "rev-parse", "HEAD"], text=True).strip(),
        study_root=str(root.resolve()), sr=None, speedup=None)
    write_json(out_dir / "report.json", evidence)
    try:
        report = json.loads((root / "report.json").read_text())
        frozen = json.loads((root / "frozen-cases.json").read_text())
        if report["status"] != "COMPLETE" or report["exit_code"] != 0:
            raise ValueError("incomplete diagnostic run")
        if report["attempted_calls"] != report["completed_calls"] or report["completed_calls"] != frozen["planned_calls"]:
            raise ValueError("diagnostic call coverage mismatch")
        if report["completed_calls"] > 49 or sha256(root / "frozen-cases.json") != report["frozen_cases_sha256"]:
            raise ValueError("frozen case identity or call cap changed")
        if sha256(root / "raw/records.jsonl") != report["raw_index_sha256"]:
            raise ValueError("diagnostic raw index hash changed")
        rows = [json.loads(line) for line in (root / "raw/records.jsonl").read_text().splitlines()]
        by_path = {row["path"]: row for row in rows}
        if (len(by_path) != len(rows) or len(rows) != report["artifacts"] or len(rows) != frozen["planned_calls"] or
            set(by_path) != {path.name for path in (root / "raw").glob("*.npz")} or
            sum(row["raw_bytes"] for row in rows) != report["raw_bytes"]):
            raise ValueError("raw diagnostic coverage or byte accounting changed")
        if report["raw_bytes"] > 16777216:
            raise ValueError("diagnostic archive exceeds its frozen size cap")
        capture_root = Path(frozen["capture_root"])
        if sha256(capture_root / "report.json") != frozen["capture_report_sha256"]:
            raise ValueError("native profile identity changed")
        capture, indexed = index_capture(capture_root)
        if capture["checkpoint_sha256"] != report["checkpoint_sha256"]:
            raise ValueError("native and diagnostic checkpoints differ")
        references = {}
        visited = set()
        for row in report["references"]:
            name = row["input_id"]
            if name in references or row["raw_and_executable_parity"] is not True:
                raise ValueError("duplicated or failed reference")
            baseline = load_record(root / "raw", by_path[row["artifact"]])
            original = load_record(capture_root / "raw", indexed[name, "reference", None, None])
            original["action"] = load_record(capture_root / "raw", indexed[name, "actions", None, None])["native"]
            if set(baseline) != set(original) or any(not np.array_equal(baseline[k], original[k]) for k in baseline):
                raise ValueError("archived native-control parity fails")
            references[name] = baseline
            visited.add(row["artifact"])
        if set(references) != set(frozen["input_ids"]):
            raise ValueError("reference observation coverage changed")
        cases = {case["case_id"]: case for case in frozen["cases"]}
        if (len(cases) != len(frozen["cases"]) or len(cases) != len(report["cases"]) or
            {case["case_id"] for case in report["cases"]} != set(cases)):
            raise ValueError("diagnostic cell coverage changed")
        with (out_dir / "case-metrics.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["case_id", "input_id", "step", "layer", "operation", "scope", "method", "bottom", "count",
                "action_relative_l2", "prefix_relative_l2", "raw_prefix_relative_l2", "prefix_gripper_disagreements", "future_relative_l2"])
            for case in sorted(report["cases"], key=lambda row: row["case_id"]):
                if any(case[k] != v for k, v in cases[case["case_id"]].items()):
                    raise ValueError("executed cell differs from frozen case")
                actual = case["actual_records"]
                if (len(actual) != 1 or actual[0]["full_projection_work_retained"] is not True or
                    any(actual[0][key] != case[key] for key in ("step", "layer", "operation", "scope", "indices", "count"))):
                    raise ValueError("native intervention trace differs from frozen case")
                if case["artifact"] in visited or sha256(root / "raw" / case["artifact"]) != case["sha256"]:
                    raise ValueError("duplicate or changed raw case")
                arrays = load_record(root / "raw", by_path[case["artifact"]])
                metrics = output_diagnostics(arrays, references[case["input_id"]])
                if metrics != case["diagnostics"]:
                    raise ValueError("saved metrics disagree with raw outputs")
                if case["scope"] == "AV" and not np.array_equal(arrays["video_latents"], references[case["input_id"]]["video_latents"]):
                    raise ValueError("AV intervention changed video")
                if case["scope"] == "VV" and (case["step"], case["layer"]) == (9, 29) and not metrics["raw_action"]["bitwise_equal"]:
                    raise ValueError("final VV intervention changed same-step action")
                visited.add(case["artifact"])
                writer.writerow([*(case[k] for k in ("case_id", "input_id", "step", "layer", "operation", "scope", "method", "bottom", "count")),
                    metrics["action"]["relative_l2"], metrics["executed_prefix"]["relative_l2"],
                    metrics["raw_prefix"]["relative_l2"], metrics["executed_prefix"]["gripper_sign_disagreements"],
                    metrics["future_latents"]["relative_l2"]])
        if visited != set(by_path):
            raise ValueError("unvalidated raw case remains")
        evidence.update(status="COMPLETE", exit_code=0, verified_files=len(visited), verified_cases=len(cases),
            study_source_commit=report["source_commit"], checkpoint_sha256=report["checkpoint_sha256"],
            study_report_sha256=sha256(root / "report.json"), frozen_cases_sha256=report["frozen_cases_sha256"],
            raw_index_sha256=report["raw_index_sha256"], completed_calls=report["completed_calls"],
            artifacts={"case-metrics.csv": sha256(out_dir / "case-metrics.csv")},
            limitations=["two correlated development observations; cases are interventions, not independent episodes",
                "metric replay checks raw-output consistency, not control success or selector speed"])
    except BaseException as exc:
        evidence.update(status="ERROR", exit_code=1, error=repr(exc))
        raise
    finally:
        evidence["end_utc"] = stamp()
        write_json(out_dir / "report.json", evidence)
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.study, args.out_dir)
    print(f"{report['status']}: {report['verified_files']} raw files, {report['verified_cases']} cases")


if __name__ == "__main__":
    main()
