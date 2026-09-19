#!/usr/bin/env python3
"""Compare first policy-input hashes before any policy-dependent action.

This audits repeatability on the intersection of recorded initial identities.
It never computes SR or equates a hash mismatch with a large image error.
Different pilot/full manifest sizes are allowed and reported explicitly.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from paired_sr import load_manifest


def read_inputs(run):
    manifest, planned = load_manifest(run)
    rows = {}
    for source in sorted(run.glob("episodes/*/attempts/*/policy-inputs.json")):
        record = json.loads((source.parent / "result.json").read_text())
        if record["status"] not in {"succeeded", "failed"}:
            continue
        identity = tuple(record[key] for key in ("task_id", "init_index", "repeat", "seed"))
        if identity not in planned or identity in rows:
            raise ValueError("out-of-manifest or duplicate settled identity")
        trace = json.loads(source.read_text())
        if not trace or trace[0]["step_index"] != manifest["protocol"]["wait_steps"]:
            raise ValueError("first policy input is missing or is not after the prescribed wait")
        rows[identity] = dict(episode_id=record["episode_id"], first=trace[0],
                              trace_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    return manifest, planned, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    left_manifest, left_plan, left = read_inputs(args.reference)
    right_manifest, right_plan, right = read_inputs(args.candidate)
    if left_manifest["protocol"] != right_manifest["protocol"]:
        raise ValueError("protocols differ")
    for key in ("backend", "suite", "root"):
        if left_manifest["benchmark"].get(key) != right_manifest["benchmark"].get(key):
            raise ValueError(f"benchmark identity differs: {key}")
    rows = []
    for identity in sorted(left.keys() & right.keys()):
        reference, candidate = left[identity]["first"], right[identity]["first"]
        rows.append(dict(identity=identity, reference=left[identity], candidate=right[identity],
                         state_equal=reference["state"] == candidate["state"],
                         agentview_equal=reference["images"]["agentview"] == candidate["images"]["agentview"],
                         wrist_equal=reference["images"]["wrist"] == candidate["images"]["wrist"]))
    result = dict(utc=datetime.now(timezone.utc).isoformat(), reference=str(args.reference),
                  candidate=str(args.candidate), reference_planned=len(left_plan), candidate_planned=len(right_plan),
                  reference_recorded=len(left), candidate_recorded=len(right), pairs=len(rows),
                  state_equal=sum(row["state_equal"] for row in rows),
                  agentview_equal=sum(row["agentview_equal"] for row in rows),
                  wrist_equal=sum(row["wrist_equal"] for row in rows), rows=rows, sr=None,
                  limitation="Hashes detect non-identity, not image-error magnitude or cause. Intersection is for input auditing only, never an SR denominator.")
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
