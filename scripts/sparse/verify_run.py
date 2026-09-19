#!/usr/bin/env python3
"""Assert that a launched run really executed the Sparse-WAM configuration it claims.

Written after a real incident: an experiment YAML requested ``sparse.enabled: true`` and the
platform faithfully passed the options to the worker, but the model checkout on the server
predated the adapter change that consumes them. Nothing failed - the run simply executed
dense and would have been reported as sparse. A requested option is not an executed option.

The adapter's ``describe()`` fingerprint is recorded in ``provenance.json`` before any
episode runs, so this check can run within a minute of launch and stop a mislabelled run
before it consumes GPU hours.

    python scripts/sparse/verify_run.py --run-dir <dir> --expect-selection av --expect-ratio 0.25
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_fingerprint(run_dir: Path) -> dict:
    provenance = run_dir / "provenance.json"
    if not provenance.is_file():
        raise SystemExit(f"no provenance.json yet under {run_dir}")
    payload = json.loads(provenance.read_text())
    worker = payload.get("policy_worker") or {}
    description = worker.get("description") or {}
    fingerprint = description.get("fingerprint")
    if not isinstance(fingerprint, dict):
        raise SystemExit(
            "provenance.json has no policy_worker.description.fingerprint; the adapter "
            "did not describe itself, so nothing about the run can be verified"
        )
    return fingerprint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--expect-enabled", action="store_true")
    parser.add_argument("--expect-selection", default=None)
    parser.add_argument("--expect-ratio", type=float, default=None)
    parser.add_argument("--expect-backend", default=None)
    parser.add_argument("--expect-checkpoint", default=None)
    args = parser.parse_args()

    fingerprint = load_fingerprint(args.run_dir)
    sparse = fingerprint.get("sparse")
    failures: list[str] = []

    if not isinstance(sparse, dict):
        failures.append(
            "fingerprint has no `sparse` block: the running adapter does not consume "
            "policy.options.sparse, so this run is DENSE regardless of its config"
        )
    else:
        if args.expect_enabled and not sparse.get("enabled"):
            failures.append(f"sparse.enabled is {sparse.get('enabled')!r}, expected true")
        if args.expect_selection and sparse.get("selection") != args.expect_selection:
            failures.append(
                f"selection is {sparse.get('selection')!r}, expected {args.expect_selection!r}"
            )
        if args.expect_ratio is not None and float(sparse.get("future_ratio", -1)) != args.expect_ratio:
            failures.append(
                f"future_ratio is {sparse.get('future_ratio')!r}, expected {args.expect_ratio}"
            )
        if args.expect_backend and sparse.get("backend") != args.expect_backend:
            failures.append(
                f"backend is {sparse.get('backend')!r}, expected {args.expect_backend!r}"
            )
    if args.expect_checkpoint:
        recorded = fingerprint.get("checkpoint")
        if recorded != args.expect_checkpoint:
            failures.append(f"checkpoint is {recorded!r}, expected {args.expect_checkpoint!r}")

    summary = {
        "run_dir": str(args.run_dir),
        "adapter": fingerprint.get("adapter"),
        "checkpoint": fingerprint.get("checkpoint"),
        "sparse": sparse,
        "sparse_config_hash": fingerprint.get("sparse_config_hash"),
        "passed": not failures,
        "failures": failures,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
