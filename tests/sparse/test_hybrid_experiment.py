"""Scientific boundaries: deterministic resume, complete pairing, no fake SR."""

import json
import importlib.util
from pathlib import Path

import pytest

from dreamwam.sparse.hybrid.experiment import read_journal, request_cells, summarize
from dreamwam.sparse.hybrid import HybridConfig, compile_plan
from test_hybrid_schedule import options


def records(cells):
    return {cell["request_id"]: dict(**cell, seconds=2.0 if cell["variant"] == "dense_strong" else 1.0,
                own_eager_parity=True, action_diagnostics=dict(relative_l2=0.1)) for cell in cells}


def test_search_order_and_resume_identity_are_deterministic(tmp_path):
    cells = request_cells(["a", "b", "c"], ["x", "y"], repeats=2, group_size=2)
    assert cells == request_cells(["a", "b", "c"], ["x", "y"], repeats=2, group_size=2)
    assert len({row["request_id"] for row in cells}) == len(cells)
    rows = records(cells)
    path = tmp_path / "requests.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in list(rows.values())[:5]))
    assert len(read_journal(path, cells)) == 5
    with path.open("a") as handle:
        handle.write(json.dumps(next(iter(rows.values()))) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        read_journal(path, cells)


def test_incomplete_comparison_never_exports_complete_speedup_or_sr():
    cells = request_cells(["a"], ["x", "y"], repeats=2, controls=("dense_strong",))
    rows = records(cells)
    complete = summarize(cells, rows, ["a"])
    assert complete["candidates"][0]["paired_speedup"] == 2
    assert complete["sr"] is None
    dense_key = next(key for key, row in rows.items() if row["variant"] == "dense_strong")
    del rows[dense_key]
    partial = summarize(cells, rows, ["a"])
    assert partial["status"] == "PARTIAL"
    assert partial["candidates"][0]["paired_speedup"] is None
    assert partial["shortlist"] == []


def test_journal_rejects_unplanned_order_nonfinite_time_and_unverified_actions(tmp_path):
    cells = request_cells(["a"], ["x"], repeats=1, controls=("dense_strong",))
    base = next(iter(records(cells).values()))
    for change in (dict(order_position=99), dict(seconds=float("nan")), dict(own_eager_parity=False)):
        path = tmp_path / "requests.jsonl"
        path.write_text(json.dumps({**base, **change}) + "\n")
        with pytest.raises(ValueError):
            read_journal(path, cells)


def test_report_exports_a_hash_checked_profile_that_compiles(tmp_path):
    config = HybridConfig.from_mapping(options())
    candidate = config.policy_hash
    cells = request_cells([candidate], ["x"], repeats=1, controls=("dense_strong",))
    rows = records(cells)
    compatibility = dict(video_length=294, tokens_per_frame=98, scheduler_hash="a" * 64)
    for row in rows.values():
        row["counters"] = dict(compatibility=compatibility)
    (tmp_path / "requests.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows.values()))
    (tmp_path / "manifest.json").write_text(json.dumps(dict(
        identity="fixture", cells=cells, candidates={candidate: config.describe()})))
    path = Path(__file__).resolve().parents[2] / "scripts/sparse/summarize_hybrid_schedules.py"
    spec = importlib.util.spec_from_file_location("hybrid_report_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.report(tmp_path, export=True)
    assert report["status"] == "COMPLETE" and report["sr"] is None
    exported = json.loads((tmp_path / "profiles" / (candidate + ".options.json")).read_text())
    loaded = HybridConfig.from_mapping(exported["hybrid_visual"])
    assert compile_plan(loaded, 10, compatibility=compatibility).plan_hash == candidate
