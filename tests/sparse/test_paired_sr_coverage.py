"""A partial/intersection-only result must never turn into a reported SR."""

import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/sparse/paired_sr.py"


def write_run(path, statuses, *, planned=3, seed=42, suite="libero_spatial"):
    path.mkdir()
    episodes = [dict(task_id=0, init_index=i, repeat=0, seed=seed) for i in range(planned)]
    (path / "manifest.json").write_text(json.dumps(dict(
        schema_version="action-eval/manifest/v1", planned_episodes=planned, episodes=episodes,
        benchmark=dict(backend="libero", suite=suite),
        protocol=dict(id="test", seed=42, replan_steps=10),
    )))
    with (path / "per_episode.csv").open("w") as handle:
        fields = ["task_id", "init_index", "repeat", "seed", "status", "success", "wall_seconds", "policy_calls", "task_name"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, status in enumerate(statuses):
            writer.writerow(dict(task_id=0, init_index=i, repeat=0, seed=seed,
                                 status=status, success=str(status == "succeeded").lower(),
                                 wall_seconds=1, policy_calls=1, task_name="test"))


def compare(tmp_path):
    result = subprocess.run([sys.executable, str(SCRIPT), "--dense", str(tmp_path / "dense"),
                             "--sparse", str(tmp_path / "sparse"), "--resamples", "50"],
                            capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


@pytest.mark.parametrize("statuses", [["succeeded", "error", "succeeded"], ["succeeded"], []])
def test_errors_missing_rows_and_empty_results_withhold_every_rate(tmp_path, statuses):
    write_run(tmp_path / "dense", ["succeeded"] * 3)
    write_run(tmp_path / "sparse", statuses)
    result = compare(tmp_path)
    assert not result["coverage"]["complete"]
    assert result["success_rate"]["dense"] is None
    assert result["success_rate"]["sparse"] is None
    assert result["success_rate"]["dense_wilson95"] is None
    assert result["paired"]["delta_ci95"] is None
    assert result["paired"]["mcnemar_one_sided_sparse_worse_p"] is None
    assert result["paired"]["dense_win_sparse_loss"] == 0, "errors must not become failures"


def test_matching_truncated_csvs_do_not_establish_complete_coverage(tmp_path):
    write_run(tmp_path / "dense", ["succeeded"])
    write_run(tmp_path / "sparse", ["succeeded"])
    assert compare(tmp_path)["success_rate"]["dense"] is None


def test_different_seed_or_manifest_subset_is_not_full_pairing(tmp_path):
    write_run(tmp_path / "dense", ["succeeded"] * 3)
    write_run(tmp_path / "sparse", ["succeeded"] * 3, seed=43)
    assert compare(tmp_path)["coverage"]["paired_terminal"] == 0
    assert compare(tmp_path)["success_rate"]["dense"] is None


def test_complete_official_outcomes_report_rates_and_correct_one_sided_tail(tmp_path):
    write_run(tmp_path / "dense", ["succeeded"] * 3)
    write_run(tmp_path / "sparse", ["failed"] * 3)
    result = compare(tmp_path)
    assert result["coverage"]["complete"]
    assert result["success_rate"]["dense"] == 1.0
    assert result["success_rate"]["sparse"] == 0.0
    assert result["paired"]["dense_win_sparse_loss"] == 3
    assert result["paired"]["mcnemar_one_sided_sparse_worse_p"] == 0.125
    assert result["paired"]["bootstrap_degenerate"] is True


def test_single_episode_stratum_withholds_false_precision_but_keeps_counts(tmp_path):
    write_run(tmp_path / "dense", ["succeeded"], planned=1)
    write_run(tmp_path / "sparse", ["failed"], planned=1)
    result = compare(tmp_path)
    assert result["coverage"]["complete"]
    assert result["success_rate"]["dense"] == 1.0
    assert result["success_rate"]["dense_wilson95"][0] < 1.0
    assert result["paired"]["delta_ci95"] is None
    assert result["paired"]["resamples"] == 0
    assert "fewer than two" in result["paired"]["bootstrap_withheld_reason"]


def test_different_suites_are_rejected(tmp_path):
    write_run(tmp_path / "dense", ["succeeded"] * 3)
    write_run(tmp_path / "sparse", ["succeeded"] * 3, suite="libero_goal")
    with pytest.raises(subprocess.CalledProcessError):
        compare(tmp_path)


def test_duplicate_rows_are_rejected(tmp_path):
    write_run(tmp_path / "dense", ["succeeded"] * 3)
    write_run(tmp_path / "sparse", ["succeeded"] * 3)
    path = tmp_path / "sparse/per_episode.csv"
    with path.open("a") as handle:
        handle.write(path.read_text().splitlines()[1] + "\n")
    with pytest.raises(subprocess.CalledProcessError):
        compare(tmp_path)
