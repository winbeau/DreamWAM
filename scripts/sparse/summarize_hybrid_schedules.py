#!/usr/bin/env python3
"""Audit a search journal, report coverage, and export replayable candidate profiles."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dreamwam.sparse.hybrid import HybridConfig
from dreamwam.sparse.hybrid.experiment import read_journal, summarize


def report(directory, *, export=False):
    manifest = json.loads((directory / "manifest.json").read_text())
    rows = read_journal(directory / "requests.jsonl", manifest["cells"])
    candidates = manifest["candidates"]
    result = summarize(manifest["cells"], rows, list(candidates))
    result["manifest_identity"] = manifest["identity"]
    # Keep the historical interval-5 point if it was in the searched universe,
    # regardless of its action error or Pareto membership.
    for candidate, payload in candidates.items():
        config = HybridConfig.from_mapping(payload)
        periodic = tuple("dense" if i == 0 else "sparse" if i % 5 == 0 else "reuse"
                         for i in range(config.schedule.num_steps))
        complete = next(r["complete"] for r in result["candidates"] if r["candidate_id"] == candidate)
        if complete and config.schedule.operations == periodic and candidate not in result["shortlist"]:
            result["shortlist"].append(candidate)
    if export:
        profiles = directory / "profiles"
        profiles.mkdir(exist_ok=True)
        for candidate in result["shortlist"]:
            config = HybridConfig.from_mapping(candidates[candidate])
            compatibilities = [row["counters"]["compatibility"] for row in rows.values()
                               if row["variant"] == candidate]
            if not compatibilities or any(c != compatibilities[0] for c in compatibilities):
                raise ValueError("cannot freeze a single-layout profile from mixed layouts")
            profile = dict(schema_version=1, num_steps=config.schedule.num_steps,
                           operations=list(config.schedule.operations), policy_hash=config.policy_hash,
                           compatibility=compatibilities[0])
            data = (json.dumps(profile, indent=2, sort_keys=True) + "\n").encode()
            path = profiles / (candidate + ".json")
            if path.exists() and path.read_bytes() != data:
                raise ValueError("refusing to overwrite a different frozen profile")
            path.write_bytes(data)
            options = config.describe()
            options["schedule"] = dict(kind="profile", num_steps=config.schedule.num_steps,
                                       path=str(path.resolve()), sha256=hashlib.sha256(data).hexdigest())
            # A policy-options fragment, not a fabricated benchmark/SR manifest.
            (profiles / (candidate + ".options.json")).write_text(json.dumps(
                dict(hybrid_visual=options, denoising_steps=config.schedule.num_steps), indent=2) + "\n")
    (directory / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--export-profiles", action="store_true")
    args = p.parse_args()
    print(json.dumps(report(args.run_dir, export=args.export_profiles), indent=2))


if __name__ == "__main__":
    main()
