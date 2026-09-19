#!/usr/bin/env python3
"""Paired Sparse-WAM vs dense SR comparison over one benchmark suite.

Consumes two ``per_episode.csv`` files produced by action-eval and reports the paired
difference, because an unpaired comparison of two success rates at ~99% has almost no
power: the interesting quantity is how many episodes changed outcome and in which
direction.

Rules this tool enforces, from the evaluation protocol:

* an episode only counts as an outcome when it reached a terminal state; ``error`` rows are
  reported separately and excluded from the rate, never folded into a failure;
* pairing includes the episode seed; equal benchmark/protocol manifests are required;
* rates and inference statistics are null unless BOTH complete planned manifests have
  terminal outcomes. Errors and missing rows never shrink the SR denominator;
* the interval is a task-stratified paired bootstrap over episodes, so the estimate is
  explicitly benchmark-conditional rather than a claim about new tasks;
* exact McNemar on the discordant pairs is reported alongside, since with one or two
  discordant pairs the bootstrap interval is degenerate and would falsely look precise.

    python scripts/sparse/paired_sr.py --dense <run-dir> --sparse <run-dir> [--json out.json]
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from pathlib import Path

TERMINAL = {"succeeded", "failed"}


def load_episodes(run_dir: Path) -> dict[tuple, dict]:
    path = run_dir / "per_episode.csv"
    if not path.is_file():
        raise FileNotFoundError(f"missing per_episode.csv under {run_dir}")
    episodes: dict[tuple, dict] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["task_id"]), int(row["init_index"]), int(row["repeat"]), int(row["seed"]))
            if key in episodes:
                raise ValueError(f"duplicate episode row in {path}: {key}")
            success = row["success"].strip().lower() == "true"
            if row["status"] in TERMINAL and success != (row["status"] == "succeeded"):
                raise ValueError(f"status/success disagree in {path}: {key}")
            episodes[key] = {
                "status": row["status"],
                "success": success,
                "wall_seconds": float(row["wall_seconds"] or 0.0),
                "policy_calls": int(row["policy_calls"] or 0),
                "task_name": row["task_name"],
                "error_type": row.get("error_type") or "",
            }
    return episodes


def load_manifest(run_dir: Path) -> tuple[dict, set[tuple]]:
    """CSV intersection alone cannot establish coverage of the planned benchmark."""
    path = run_dir / "manifest.json"
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != "action-eval/manifest/v1":
        raise ValueError(f"unsupported or missing manifest schema: {path}")
    entries = manifest.get("episodes", [])
    keys = {(int(row["task_id"]), int(row["init_index"]), int(row["repeat"]), int(row["seed"]))
            for row in entries}
    if not keys or len(keys) != len(entries):
        raise ValueError(f"empty or duplicate planned episode identities: {path}")
    if manifest.get("planned_episodes") != len(keys):
        raise ValueError(f"planned_episodes disagrees with manifest entries: {path}")
    if not manifest.get("protocol") or not manifest.get("benchmark", {}).get("suite"):
        raise ValueError(f"manifest lacks benchmark/protocol identity: {path}")
    return manifest, keys


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return (float("nan"), float("nan"))
    phat = successes / total
    denominator = 1 + z * z / total
    center = (phat + z * z / (2 * total)) / denominator
    half = (
        z * ((phat * (1 - phat) / total + z * z / (4 * total * total)) ** 0.5) / denominator
    )
    return (max(0.0, center - half), min(1.0, center + half))


def mcnemar_exact(b: int, c: int) -> tuple[float, float]:
    """Two-sided and one-sided (sparse worse) exact McNemar p-values."""
    from math import comb

    n = b + c
    if n == 0:
        return (1.0, 1.0)
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2**n)
    two_sided = min(1.0, 2 * tail)
    # b counts Dense wins / Sparse losses: evidence that Sparse is worse is the
    # UPPER tail for b, not its lower tail (which would approach one as b grows).
    one_sided = sum(comb(n, i) for i in range(b, n + 1)) / (2**n)
    return (two_sided, one_sided)


def stratified_bootstrap(
    per_task: dict[int, list[tuple[bool, bool]]],
    *,
    resamples: int,
    seed: int,
) -> dict:
    """Task-stratified paired bootstrap over episodes, tasks equally weighted."""
    rng = random.Random(seed)
    tasks = sorted(per_task)
    deltas: list[float] = []
    dense_rates: list[float] = []
    sparse_rates: list[float] = []
    for _ in range(resamples):
        task_deltas = []
        task_dense = []
        task_sparse = []
        for task in tasks:
            pairs = per_task[task]
            n = len(pairs)
            sample = [pairs[rng.randrange(n)] for _ in range(n)]
            task_deltas.append(
                statistics.fmean(
                    (1.0 if s else 0.0) - (1.0 if d else 0.0) for d, s in sample
                )
            )
            task_dense.append(statistics.fmean(1.0 if d else 0.0 for d, _ in sample))
            task_sparse.append(statistics.fmean(1.0 if s else 0.0 for _, s in sample))
        deltas.append(statistics.fmean(task_deltas))
        dense_rates.append(statistics.fmean(task_dense))
        sparse_rates.append(statistics.fmean(task_sparse))

    def interval(values: list[float]) -> tuple[float, float]:
        ordered = sorted(values)
        low = ordered[int(0.025 * (len(ordered) - 1))]
        high = ordered[int(0.975 * (len(ordered) - 1))]
        return (low, high)

    return {
        "delta_ci95": interval(deltas),
        "dense_sr_ci95": interval(dense_rates),
        "sparse_sr_ci95": interval(sparse_rates),
        "resamples": resamples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense", required=True, type=Path)
    parser.add_argument("--sparse", required=True, type=Path)
    parser.add_argument("--label", default="sparse")
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    dense = load_episodes(args.dense)
    sparse = load_episodes(args.sparse)
    dense_manifest, dense_expected = load_manifest(args.dense)
    sparse_manifest, sparse_expected = load_manifest(args.sparse)
    if dense_manifest["benchmark"] != sparse_manifest["benchmark"]:
        raise ValueError("cannot pair different benchmark identities")
    if dense_manifest["protocol"] != sparse_manifest["protocol"]:
        raise ValueError("cannot pair different protocol identities")
    if set(dense) - dense_expected or set(sparse) - sparse_expected:
        raise ValueError("CSV contains episodes outside its planned manifest")
    shared = sorted(set(dense) & set(sparse))

    paired = [key for key in shared if dense[key]["status"] in TERMINAL and sparse[key]["status"] in TERMINAL]
    excluded_errors = [
        key
        for key in shared
        if dense[key]["status"] not in TERMINAL or sparse[key]["status"] not in TERMINAL
    ]

    b = c = 0
    per_task: dict[int, list[tuple[bool, bool]]] = {}
    dense_wall: list[float] = []
    sparse_wall: list[float] = []
    for key in paired:
        d = dense[key]["success"]
        s = sparse[key]["success"]
        per_task.setdefault(key[0], []).append((d, s))
        if d and not s:
            b += 1
        elif s and not d:
            c += 1
        dense_wall.append(dense[key]["wall_seconds"])
        sparse_wall.append(sparse[key]["wall_seconds"])

    n = len(paired)
    complete = (dense_expected == sparse_expected and set(dense) == dense_expected
                and set(sparse) == sparse_expected and n == len(dense_expected))
    dense_success = sum(1 for key in paired if dense[key]["success"])
    sparse_success = sum(1 for key in paired if sparse[key]["success"])
    two_sided, one_sided = mcnemar_exact(b, c) if complete else (None, None)
    bootstrap = (stratified_bootstrap(per_task, resamples=args.resamples, seed=args.seed)
                 if complete else dict(delta_ci95=None, dense_sr_ci95=None,
                                       sparse_sr_ci95=None, resamples=0))

    report = {
        "tool": "scripts/sparse/paired_sr.py",
        "label": args.label,
        "dense_run": str(args.dense),
        "sparse_run": str(args.sparse),
        "coverage": {
            "complete": complete,
            "expected_dense_episodes": len(dense_expected),
            "expected_sparse_episodes": len(sparse_expected),
            "planned_identity_difference": len(dense_expected ^ sparse_expected),
            "missing_dense_rows": len(dense_expected - set(dense)),
            "missing_sparse_rows": len(sparse_expected - set(sparse)),
            "dense_episodes": len(dense),
            "sparse_episodes": len(sparse),
            "shared_episodes": len(shared),
            "paired_terminal": n,
            "excluded_non_terminal": len(excluded_errors),
            "unpaired": len(set(dense) ^ set(sparse)),
            "sparse_non_terminal_detail": {
                status: sum(
                    1 for key in shared if sparse[key]["status"] == status
                )
                for status in sorted({sparse[key]["status"] for key in shared})
                if status not in TERMINAL
            },
        },
        "success_rate": {
            "dense": dense_success / n if complete else None,
            "sparse": sparse_success / n if complete else None,
            "delta_absolute": (sparse_success - dense_success) / n if complete else None,
            "dense_wilson95": wilson(dense_success, n) if complete else None,
            "sparse_wilson95": wilson(sparse_success, n) if complete else None,
            "withheld_reason": None if complete else "incomplete or different planned episode coverage",
        },
        "paired": {
            "dense_win_sparse_loss": b,
            "dense_loss_sparse_win": c,
            "unchanged": n - b - c,
            "mcnemar_two_sided_p": two_sided,
            "mcnemar_one_sided_sparse_worse_p": one_sided,
            **bootstrap,
        },
        "per_task": {
            str(task): {
                "pairs": len(pairs),
                "dense_success": sum(1 for d, _ in pairs if d),
                "sparse_success": sum(1 for _, s in pairs if s),
                "dense_win_sparse_loss": sum(1 for d, s in pairs if d and not s),
                "dense_loss_sparse_win": sum(1 for d, s in pairs if s and not d),
            }
            for task, pairs in sorted(per_task.items())
        },
        "timing_seconds": {
            "dense_mean": statistics.fmean(dense_wall) if dense_wall else None,
            "sparse_mean": statistics.fmean(sparse_wall) if sparse_wall else None,
            "note": (
                "wall time includes simulator stepping and render aborts, so it is not the "
                "model's own latency; the operator-level cost is measured separately"
            ),
        },
    }

    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    print(text)
    if args.json:
        args.json.write_text(text + "\n")


if __name__ == "__main__":
    main()
