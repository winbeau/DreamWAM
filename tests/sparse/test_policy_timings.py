import importlib.util
import json
from pathlib import Path
import sys

import pytest


@pytest.fixture
def timings(monkeypatch):
    scripts = Path(__file__).resolve().parents[2] / "scripts/sparse"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("policy_timings_test", scripts / "summarize_policy_timings.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_run(root, seconds, *, complete=True, prompt_hits=(False, True, True)):
    root.mkdir()
    episode = dict(task_id=0, init_index=0, repeat=0, seed=42)
    manifest = dict(schema_version="action-eval/manifest/v1", planned_episodes=1,
                    benchmark={"suite": "libero_spatial"}, protocol={"max_steps": 400}, episodes=[episode])
    (root / "manifest.json").write_text(json.dumps(manifest))
    if not complete:
        return
    directory = root / "episodes/e/attempts/0001"
    directory.mkdir(parents=True)
    calls = [dict(call_index=i, transport_seconds=t + .01, metadata={
        "timing": {"predict_seconds": t}, "diagnostics": {"prompt_cache": {"last_hit": hit}}})
        for i, (t, hit) in enumerate(zip(seconds, prompt_hits), 1)]
    path = directory / "policy-calls.json"
    path.write_text(json.dumps(calls))
    result = dict(**episode, episode_id="e", status="succeeded", policy_calls=len(calls),
        policy_seconds=sum(t + .01 for t in seconds), wall_seconds=10, env_seconds=5,
        artifacts={"policy_calls": str(path.relative_to(root))})
    (directory / "result.json").write_text(json.dumps(result))


def test_warm_cold_and_transport_are_separate(timings, tmp_path):
    dense, sparse = tmp_path / "dense", tmp_path / "sparse"
    make_run(dense, [2, .3, .3])
    make_run(sparse, [4, .15, .15])
    result = timings.report(dense, sparse)
    assert result["complete"]
    assert result["descriptive_warm_model_speedup"] == 2
    assert result["descriptive_episode_balanced_warm_model_speedup"] == 2
    assert result["descriptive_all_model_speedup"] < 1
    assert result["descriptive_warm_transport_speedup"] != 2
    assert result["dense"]["model_seconds_first_per_episode"]["mean"] == 2


def test_incomplete_pair_has_no_speedup(timings, tmp_path):
    dense, sparse = tmp_path / "dense", tmp_path / "sparse"
    make_run(dense, [2, .3, .3])
    make_run(sparse, [], complete=False)
    result = timings.report(dense, sparse)
    assert not result["complete"]
    assert result["descriptive_warm_model_speedup"] is None


def test_changed_protocol_is_not_paired(timings, tmp_path):
    dense, sparse = tmp_path / "dense", tmp_path / "sparse"
    make_run(dense, [2, .3, .3])
    make_run(sparse, [2, .3, .3])
    path = sparse / "manifest.json"
    payload = json.loads(path.read_text())
    payload["protocol"]["max_steps"] = 200
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="different paired"):
        timings.report(dense, sparse)


def test_missing_cache_hits_do_not_become_warm(timings, tmp_path):
    dense, sparse = tmp_path / "dense", tmp_path / "sparse"
    for root in (dense, sparse):
        make_run(root, [2, .3, .3], prompt_hits=(False, False, False))
    assert timings.report(dense, sparse)["descriptive_warm_model_speedup"] is None
