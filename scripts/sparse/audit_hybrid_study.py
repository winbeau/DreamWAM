#!/usr/bin/env python3
"""Re-audit accepted timing bytes and produce a budget/selector/step result table.

No timing exclusions or candidate selection. Incomplete runs fail; official SR
stays in the evaluator's separate paired report. Inputs within an episode are
correlated and repetitions of a deterministic model are not quality samples.
"""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from dreamwam.sparse.hybrid.experiment import read_journal, summarize
from dreamwam.sparse.hybrid.schedule import stable_hash


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    journal = read_journal(directory / "requests.jsonl", manifest["cells"])
    summary = summarize(manifest["cells"], journal, list(manifest["candidates"]))
    if manifest["status"] != "COMPLETE" or summary["status"] != "COMPLETE":
        raise ValueError("require complete timing coverage")
    references = {}
    for identity, reference in manifest["references"].items():
        path = directory / reference["path"]
        if sha256(path) != reference["sha256"]:
            raise ValueError("eager reference hash mismatch")
        references[identity] = np.load(path, allow_pickle=False)
    for row in journal.values():
        path = directory / row["actions_path"]
        if sha256(path) != row["actions_sha256"]:
            raise ValueError("accepted action artifact hash mismatch")
        actual = np.load(path, allow_pickle=False)
        expected = references[stable_hash([row["variant"], row["input_id"]])]
        if actual.dtype != expected.dtype or actual.shape != expected.shape or actual.tobytes() != expected.tobytes():
            raise ValueError("action bytes differ from own eager, including signed zeros")
        if row["counters"]["action_layer_updates"] != 300:
            raise ValueError("unchanged ten-step thirty-layer action execution was not recorded")
    rows = []
    for candidate in summary["candidates"]:
        identity = candidate["candidate_id"]
        config = manifest["candidates"][identity]
        measured = [r for r in journal.values() if r["variant"] == identity]
        prefixes = []
        for input_id in sorted({r["input_id"] for r in measured}):
            actual = references[stable_hash([identity, input_id])][:10].astype(np.float64)
            dense = references[stable_hash(["dense_strong", input_id])][:10].astype(np.float64)
            prefixes.append(float(np.linalg.norm(actual - dense) / max(np.linalg.norm(dense), 1e-12)))
        rows.append(dict(run=directory.name, candidate_id=identity,
            read_ratio=config["read"]["keep_ratio"], recompute_ratio=config["recompute"]["keep_ratio"],
            selection=config["selection"]["method"], reuse_mode=config.get("reuse", {}).get("mode", "features"),
            sparse_steps=[i for i, op in enumerate(config["schedule"]["operations"]) if op == "sparse"],
            warm_ms=candidate["seconds"]["mean"] * 1000, paired_speedup=candidate["paired_speedup"],
            action_relative_l2=candidate["mean_action_relative_l2"], prefix10_relative_l2=statistics.fmean(prefixes),
            inputs=len(prefixes), samples=len(measured), **candidate["pairing_attempts"]))
    return dict(run=str(directory), source=manifest["identity_data"]["git"],
        identity=manifest["identity"], manifest_sha256=sha256(directory / "manifest.json"),
        journal_sha256=sha256(directory / "requests.jsonl"), audited_requests=len(journal),
        bitwise_eager_actions=len(journal), rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.csv.exists():
        parser.error("refusing to overwrite an existing audit")
    if len(args.runs) != len(set(args.runs)):
        parser.error("duplicate run directory")
    reports = [audit(path) for path in args.runs]
    result = dict(status="VERIFIED", runs=reports,
        audited_requests=sum(r["audited_requests"] for r in reports),
        candidate_cases=sum(len(r["rows"]) for r in reports), sr=None,
        limitations=["quality diagnostics, not SR", "same episodes are not independent samples",
                     "shared hardware; no timing outlier removed"])
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    rows = [row for report in reports for row in report["rows"]]
    with args.csv.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({key: value for key, value in result.items() if key != "runs"}))


if __name__ == "__main__":
    main()
