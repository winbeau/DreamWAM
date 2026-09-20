import importlib.util
from pathlib import Path

import numpy as np
import pytest

from dreamwam.sparse.profile.archive import sha256


@pytest.fixture
def auditor():
    path = Path(__file__).resolve().parents[2] / "scripts/sparse/audit_native_screen.py"
    spec = importlib.util.spec_from_file_location("native_screen_auditor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_raw_screen_audit_rejects_changed_bytes_and_path_escape(auditor, tmp_path):
    (tmp_path / "actions").mkdir()
    path = tmp_path / "actions/one.npz"
    np.savez_compressed(path, action=np.zeros((32, 7), dtype=np.float32),
                        raw_action=np.zeros((1, 32, 7), dtype=np.float32))
    record = dict(actions_path="actions/one.npz", actions_sha256=sha256(path))
    assert auditor.raw_output(tmp_path, record)["raw_action"].shape == (1, 32, 7)
    with pytest.raises(ValueError, match="escapes"):
        auditor.raw_output(tmp_path, dict(record, actions_path="../outside.npz"))
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        auditor.raw_output(tmp_path, record)


@pytest.mark.parametrize("kind", ["shape", "dtype", "nonfinite"])
def test_raw_screen_audit_rejects_invalid_output_contract(auditor, tmp_path, kind):
    (tmp_path / "actions").mkdir()
    action = np.zeros((32, 7), dtype=np.float32)
    if kind == "shape":
        action = action[:10]
    elif kind == "dtype":
        action = action.astype(np.float64)
    else:
        action[0, 0] = np.nan
    path = tmp_path / "actions/one.npz"
    np.savez_compressed(path, action=action, raw_action=np.zeros((1, 32, 7), dtype=np.float32))
    with pytest.raises(ValueError, match="contract"):
        auditor.raw_output(tmp_path, dict(actions_path="actions/one.npz", actions_sha256=sha256(path)))
