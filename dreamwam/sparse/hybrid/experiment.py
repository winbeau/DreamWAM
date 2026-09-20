"""Dependency-free request identities, resume audits, and paired timing reports."""

from collections import defaultdict
import json
import math
import random
import statistics

from .schedule import integer, stable_hash


def request_cells(candidate_ids, input_ids, *, repeats, group_size=2,
                  controls=("dense_strong", "legacy", "fresh"), seed=42):
    integer(repeats, "repeats")
    integer(group_size, "group_size")
    for items in (candidate_ids, input_ids, controls):
        if not items or len(set(items)) != len(items):
            raise ValueError("candidate/input/control identities must be nonempty and unique")
    if "dense_strong" not in controls or set(candidate_ids) & set(controls):
        raise ValueError("require a distinct dense_strong control")
    candidates = list(candidate_ids)
    rng = random.Random(seed)
    rng.shuffle(candidates)
    cells = []
    for group, offset in enumerate(range(0, len(candidates), group_size)):
        variants = list(controls) + candidates[offset:offset + group_size]
        rng.shuffle(variants)
        for repeat in range(repeats):
            for input_index, input_id in enumerate(input_ids):
                shift = (repeat * len(input_ids) + input_index) % len(variants)
                order = variants[shift:] + variants[:shift]
                for position, variant in enumerate(order):
                    identity = dict(group=group, repeat=repeat, input_id=input_id, variant=variant)
                    cells.append(dict(**identity, request_id=stable_hash(identity),
                                      order=order, order_position=position))
    return cells


def read_journal(path, cells):
    planned = {cell["request_id"]: cell for cell in cells}
    rows = {}
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        row = json.loads(line)  # Corruption is an error, never silently discarded.
        key = row["request_id"]
        if key not in planned or key in rows:
            raise ValueError("unplanned or duplicate request in journal")
        if any(row.get(name) != value for name, value in planned[key].items()):
            raise ValueError("request order/identity differs from frozen manifest")
        if isinstance(row["seconds"], bool) or not isinstance(row["seconds"], (float, int)):
            raise ValueError("timing must be a number")
        if not math.isfinite(row["seconds"]) or row["seconds"] <= 0:
            raise ValueError("invalid request timing")
        if row.get("own_eager_parity") is not True:
            raise ValueError("request has no validated own-eager reference")
        if "attempt_id" in row:
            integer(row["attempt_id"], "attempt_id", 0)
        rows[key] = row
    return rows


def distribution(values):
    if not values:
        return None
    values = sorted(values)
    def percentile(p):
        position = (len(values) - 1) * p
        low, high = math.floor(position), math.ceil(position)
        return values[low] + (values[high] - values[low]) * (position - low)
    return dict(samples=len(values), mean=statistics.fmean(values),
                p50=percentile(0.5), p95=percentile(0.95),
                min=values[0], max=values[-1])


def summarize(cells, rows, candidate_ids):
    by_variant = defaultdict(list)
    lookup = {}
    for row in rows.values():
        by_variant[row["variant"]].append(row)
        lookup[row["group"], row["repeat"], row["input_id"], row["variant"]] = row
    results = []
    controls = sorted({cell["variant"] for cell in cells} - set(candidate_ids))
    for candidate in candidate_ids:
        measured = by_variant[candidate]
        expected = sum(cell["variant"] == candidate for cell in cells)
        paired = [(row, lookup.get((row["group"], row["repeat"], row["input_id"], "dense_strong")))
                  for row in measured]
        paired = [(row, dense) for row, dense in paired if dense is not None]
        sparse_time = statistics.fmean(row["seconds"] for row, _ in paired) if paired else None
        dense_time = statistics.fmean(row["seconds"] for _, row in paired) if paired else None
        complete = len(measured) == expected and len(paired) == expected
        same_attempt = [(row, dense) for row, dense in paired if row.get("attempt_id") is not None
                        and row.get("attempt_id") == dense.get("attempt_id")]
        cross_attempt = sum(row.get("attempt_id") is not None and dense.get("attempt_id") is not None
                            and row["attempt_id"] != dense["attempt_id"] for row, dense in paired)
        versus_controls = {}
        for name in controls:
            matches = [(row, lookup.get((row["group"], row["repeat"], row["input_id"], name)))
                       for row in measured]
            matches = [(row, control) for row, control in matches if control is not None]
            versus_controls[name] = (statistics.fmean(control["seconds"] for _, control in matches)
                                      / statistics.fmean(row["seconds"] for row, _ in matches)
                                      if len(matches) == expected else None)
        results.append(dict(candidate_id=candidate, complete=complete,
            coverage=dict(completed=len(measured), planned=expected, paired=len(paired)),
            seconds=distribution([r["seconds"] for r in measured]),
            paired_speedup=dense_time / sparse_time if complete else None,
            pairing_attempts=dict(same=len(same_attempt), cross=cross_attempt,
                                  untracked=len(paired) - len(same_attempt) - cross_attempt),
            paired_speedup_same_attempt=(
                statistics.fmean(dense["seconds"] for _, dense in same_attempt)
                / statistics.fmean(row["seconds"] for row, _ in same_attempt) if same_attempt else None),
            paired_speedup_vs_controls=versus_controls,
            mean_action_relative_l2=statistics.fmean(r["action_diagnostics"]["relative_l2"] for r in measured)
                if measured else None,
            sr=None))
    eligible = [row for row in results if row["complete"]]
    frontier = []
    for row in eligible:
        def dominates(other):
            a = (other["seconds"]["mean"], other["mean_action_relative_l2"])
            b = (row["seconds"]["mean"], row["mean_action_relative_l2"])
            return a[0] <= b[0] and a[1] <= b[1] and a != b
        if not any(dominates(other) for other in eligible):
            frontier.append(row["candidate_id"])
    shortlist = []
    if eligible:
        shortlist = list(dict.fromkeys([
            min(eligible, key=lambda r: r["seconds"]["mean"])["candidate_id"],
            min(eligible, key=lambda r: r["mean_action_relative_l2"])["candidate_id"]]))
    return dict(status="COMPLETE" if len(rows) == len(cells) else "PARTIAL",
                coverage=dict(completed=len(rows), planned=len(cells)),
                candidates=results, diagnostic_pareto=frontier, shortlist=shortlist,
                sr=None, scope="open-loop latency/action diagnostics; pooled timings may span attempts; no SR claim",
                controls={name: distribution([row["seconds"] for row in entries])
                          for name, entries in by_variant.items() if name not in candidate_ids})
