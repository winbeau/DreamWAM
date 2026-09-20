import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


def module():
    scripts = Path(__file__).resolve().parents[2] / "scripts/sparse"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location("trajectory_export", scripts / "export_trajectory_inputs.py")
        result = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(result)
        return result
    finally:
        sys.path.remove(str(scripts))


def fixture_run(root):
    root.mkdir()
    episodes = [dict(task_id=i, init_index=1, repeat=0, seed=42) for i in range(2)]
    (root / "manifest.json").write_text(json.dumps(dict(schema_version="action-eval/manifest/v1",
        planned_episodes=2, episodes=episodes, protocol={"wait_steps": 30}, benchmark={"suite": "libero_spatial"})))
    for i, episode in enumerate(episodes):
        directory = root / f"episodes/e{i}/attempts/0001"
        directory.mkdir(parents=True)
        (directory / "result.json").write_text(json.dumps(dict(**episode,
            status="succeeded" if i == 0 else "failed", policy_calls=5)))
        rows, hashes = [], []
        for index in range(5):
            state = np.full(8, index, dtype=np.float32)
            images = {name: np.full((4, 4, 3), index, dtype=np.uint8) for name in ("agentview", "wrist")}
            path = directory / f"{index}.npz"
            np.savez_compressed(path, state=state, instruction=np.asarray("move"), step_index=30 + index * 10,
                                **{"image_" + k: v for k, v in images.items()})
            rows.append(dict(call_index=index + 1, step_index=30 + index * 10,
                path=str(path.relative_to(root)), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            describe = lambda a: dict(sha256=hashlib.sha256(a.tobytes()).hexdigest())
            hashes.append(dict(step_index=30 + index * 10, instruction="move", state=describe(state),
                               images={key: describe(value) for key, value in images.items()}))
        (directory / "policy-observations.json").write_text(json.dumps(rows))
        (directory / "policy-inputs.json").write_text(json.dumps(hashes))


def test_export_includes_failures_and_freezes_first_middle_last_without_labels(tmp_path):
    fixture_run(tmp_path / "run")
    result = module().export(tmp_path / "run", tmp_path / "export")
    assert result["status"] == "CAPTURED" and result["split_role"] == "development"
    assert len(result["inputs"]) == 6 and result["sr"] is None
    assert [row["call_index"] for row in result["inputs"]] == [1, 3, 5] * 2
    for row in result["inputs"]:
        with np.load(tmp_path / "export" / row["path"], allow_pickle=False) as raw:
            assert set(raw) == {"state", "instruction", "agentview", "wrist"}
    with pytest.raises(FileExistsError):
        module().export(tmp_path / "run", tmp_path / "export")


def test_export_refuses_incomplete_coverage_and_input_tampering(tmp_path):
    fixture_run(tmp_path / "run")
    path = tmp_path / "run/episodes/e1/attempts/0001/result.json"
    original = path.read_text()
    path.write_text(original.replace('"failed"', '"error"'))
    with pytest.raises(ValueError, match="full planned"):
        module().export(tmp_path / "run", tmp_path / "incomplete")
    path.write_text(original)
    trace = tmp_path / "run/episodes/e1/attempts/0001/policy-inputs.json"
    trace.write_text(trace.read_text().replace('"move"', '"wrong"'))
    with pytest.raises(ValueError, match="instruction"):
        module().export(tmp_path / "run", tmp_path / "tampered")
    assert json.loads((tmp_path / "tampered/manifest.json").read_text())["status"] == "ERROR"
