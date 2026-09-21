#!/usr/bin/env python3
"""Freeze early/middle/late policy observations from COMPLETE development runs.

All terminal outcomes, including failures, are included. No success labels or
teacher actions enter the exported policy observations. The split is exposed
development, never confirmation; samples within an episode are correlated.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from paired_sr import load_manifest


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(run, out, *, all_calls=False):
    manifest, planned = load_manifest(run)
    sources = {}
    for path in sorted(run.glob("episodes/*/attempts/*/policy-observations.json")):
        record = json.loads((path.parent / "result.json").read_text())
        if record["status"] not in ("succeeded", "failed"):
            continue
        identity = tuple(record[key] for key in ("task_id", "init_index", "repeat", "seed"))
        if identity not in planned or identity in sources:
            raise ValueError("out-of-manifest or duplicate terminal observation source")
        rows = json.loads(path.read_text())
        if len(rows) != record["policy_calls"] or not rows:
            raise ValueError("incomplete policy observation trace")
        sources[identity] = (path, rows)
    if set(sources) != planned:
        raise ValueError("require full planned coverage; never select only successful episodes")
    out.mkdir(parents=True, exist_ok=False)
    result = dict(status="RUNNING", created_utc=datetime.now(timezone.utc).isoformat(),
        split_role="development", source_run=str(run), source_manifest_sha256=sha256(run / "manifest.json"),
        episode_split_sha256=hashlib.sha256(json.dumps(sorted(planned)).encode()).hexdigest(),
        protocol=manifest["protocol"], suite=manifest["benchmark"]["suite"],
        selection=("all contiguous policy calls; all terminal episodes regardless of success" if all_calls else
                   "first, floor((calls-1)/2), last; all terminal episodes regardless of success"),
        complete_call_history=all_calls,
        limitations=["observations within an episode are correlated", "exposed development, not confirmation"],
        inputs=[], sr=None)
    try:
        for identity, (trace_path, rows) in sorted(sources.items()):
            expected = json.loads((trace_path.parent / "policy-inputs.json").read_text())
            if len(expected) != len(rows):
                raise ValueError("observation/hash trace lengths disagree")
            indices = range(len(rows)) if all_calls else sorted({0, (len(rows) - 1) // 2, len(rows) - 1})
            for index in indices:
                entry = rows[index]
                if entry.get("call_index") != index + 1:
                    raise ValueError("source observation call order is incomplete")
                path = run / entry["path"]
                if sha256(path) != entry["sha256"]:
                    raise ValueError("source observation archive hash mismatch")
                with np.load(path, allow_pickle=False) as raw:
                    state = raw["state"].copy()
                    instruction = raw["instruction"].copy()
                    images = {key: raw["image_" + key].copy() for key in ("agentview", "wrist")}
                    if int(raw["step_index"]) != expected[index]["step_index"]:
                        raise ValueError("source observation step mismatch")
                for value, description in [(state, expected[index]["state"]),
                        *((images[key], expected[index]["images"][key]) for key in images)]:
                    if hashlib.sha256(value.tobytes(order="C")).hexdigest() != description["sha256"]:
                        raise ValueError("source observation differs from actual policy-input hash")
                if str(instruction.item()) != expected[index]["instruction"]:
                    raise ValueError("source instruction mismatch")
                name = "t%03d-i%03d-r%02d-s%d-c%04d" % (*identity, index + 1)
                target = out / (name + ".npz")
                np.savez_compressed(target, state=state, instruction=instruction, **images)
                result["inputs"].append(dict(id=name, path=target.name, sha256=sha256(target),
                    task_id=identity[0], init_index=identity[1], repeat=identity[2], seed=identity[3],
                    call_index=index + 1, step_index=entry["step_index"], source_path=str(path),
                    source_sha256=entry["sha256"], source_trace_sha256=sha256(trace_path)))
        result["status"] = "CAPTURED"
        return result
    except BaseException as exc:
        result.update(status="ERROR", error=repr(exc))
        raise
    finally:
        (out / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--all-calls", action="store_true", help="preserve causal contiguous history for M1")
    args = parser.parse_args()
    result = export(args.run, args.out_dir, all_calls=args.all_calls)
    print(json.dumps(dict(status=result["status"], inputs=len(result["inputs"]), sr=None)))


if __name__ == "__main__":
    main()
