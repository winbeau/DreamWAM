"""Finite schedule enumeration and strict, replayable JSONL manifests."""

from __future__ import annotations

from dataclasses import replace
from itertools import combinations, product
import json
import math

from .config import HybridConfig
from .schedule import Schedule, integer, mapping


def parse_indices(value):
    """Comma-separated integers and half-open ranges, rejecting duplicates."""
    if not value.strip():
        return ()
    result = []
    for part in value.split(","):
        if ":" in part:
            start, end = map(int, part.split(":"))
            if end <= start:
                raise ValueError("ranges must be increasing")
            result.extend(range(start, end))
        else:
            result.append(int(part))
    if len(result) != len(set(result)) or any(i < 0 for i in result):
        raise ValueError("indices must be unique and non-negative")
    return tuple(sorted(result))


def generate_candidates(config, dense_steps, sparse_steps, refresh_counts):
    n = config.schedule.num_steps
    for values in (dense_steps, sparse_steps, refresh_counts):
        if len(values) != len(set(values)):
            raise ValueError("duplicate candidate indices/counts")
        for value in values:
            integer(value, "candidate index/count", 0)
    if 0 not in dense_steps or set(dense_steps) & set(sparse_steps):
        raise ValueError("step 0 must be dense and dense/sparse candidates must be disjoint")
    if any(i >= n for i in (*dense_steps, *sparse_steps)):
        raise ValueError("candidate step outside schedule")
    if any(k > len(sparse_steps) for k in refresh_counts):
        raise ValueError("refresh count exceeds candidate step count")
    for count in sorted(refresh_counts):
        for selected in combinations(sorted(sparse_steps), count):
            operations = tuple("dense" if i in dense_steps else "sparse" if i in selected else "reuse"
                               for i in range(n))
            candidate = replace(config, schedule=Schedule(n, operations))
            yield dict(schema_version=1, candidate_id=candidate.policy_hash,
                       sparse_steps=list(selected), options=candidate.describe())


def generate_sweep(config, dense_steps, sparse_steps, refresh_counts, *,
                   read_ratios=None, recompute_ratios=None, selections=None,
                   max_candidates=512):
    """A bounded Cartesian grid; invalid/duplicate cells fail instead of disappearing.

    Budget/selector axes are independent of schedule enumeration. The resulting
    manifest uses the existing canonical identities and can be replayed/exported
    by the same benchmark without putting any search logic in the online policy.
    """
    integer(max_candidates, "max_candidates")
    axes = []
    for name, values, default in (("read_ratios", read_ratios, config.read_ratio),
                                  ("recompute_ratios", recompute_ratios, config.recompute_ratio),
                                  ("selections", selections, config.selection)):
        values = (default,) if values is None else tuple(values)
        if not values or len(values) != len(set(values)):
            raise ValueError(f"{name} must be nonempty and unique")
        axes.append(values)
    # Validate every schedule index/count before computing its cardinality.
    if next(generate_candidates(config, dense_steps, sparse_steps, refresh_counts), None) is None:
        raise ValueError("empty schedule search")
    schedule_count = sum(math.comb(len(sparse_steps), k) for k in refresh_counts)
    count = math.prod(len(axis) for axis in axes) * schedule_count
    if count > max_candidates:
        raise ValueError(f"grid has {count} candidates, exceeds max_candidates={max_candidates}")
    rows = []
    for read_ratio, recompute_ratio, selection in product(*axes):
        variant = replace(config, read_ratio=read_ratio, recompute_ratio=recompute_ratio,
                          selection=selection)
        rows.extend(generate_candidates(variant, dense_steps, sparse_steps, refresh_counts))
    if len({row["candidate_id"] for row in rows}) != count:
        raise ValueError("grid contains duplicate canonical candidates")
    return rows


def read_candidates(path):
    candidates = []
    seen = set()
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        row = mapping(json.loads(line), ("schema_version", "candidate_id", "sparse_steps", "options"),
                      f"candidate line {line_number}",
                      ("schema_version", "candidate_id", "sparse_steps", "options"))
        config = HybridConfig.from_mapping(row["options"])
        if type(row["schema_version"]) is not int or row["schema_version"] != 1:
            raise ValueError("candidate schema mismatch")
        if config.policy_hash != row["candidate_id"] or row["candidate_id"] in seen:
            raise ValueError("candidate hash mismatch or duplicate")
        if row["sparse_steps"] != [i for i, op in enumerate(config.schedule.operations) if op == "sparse"]:
            raise ValueError("sparse_steps does not match actual operations")
        seen.add(row["candidate_id"])
        candidates.append((row["candidate_id"], config))
    if not candidates:
        raise ValueError("empty candidate manifest")
    return candidates
