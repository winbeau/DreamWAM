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


def test_trace_audit_rejects_a_new_read_with_no_recomputation(auditor):
    from dreamwam.sparse.hybrid.native_experiment import native_candidates
    from dreamwam.sparse.hybrid.schedule import stable_hash
    config = dict(native_candidates("refresh"))["context_sparse_1"]
    old = list(range(19)) + list(range(98, 117)) + list(range(196, 214))
    new = sorted(set(old) - {18} | {30})
    query = list(range(10)) + list(range(98, 108)) + list(range(196, 206))
    steps = [dict(effective_op=op, route=old if i == 0 else new,
                  route_hash=stable_hash(old if i == 0 else new), query=query if i == 1 else None)
             for i, op in enumerate(config.schedule.operations)]
    with pytest.raises(ValueError, match="not freshly recomputed"):
        auditor.audit_route_trace(config, {"steps": steps})
    steps[1]["query"] = sorted(set(query) - {9} | {30})
    assert auditor.audit_route_trace(config, {"steps": steps}) == 10
