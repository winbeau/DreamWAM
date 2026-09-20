import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from dreamwam.sparse.hybrid import HybridConfig
from dreamwam.sparse.hybrid.experiment import request_cells
from dreamwam.sparse.hybrid.schedule import stable_hash
from dreamwam.sparse.hybrid.search import generate_candidates
from test_hybrid_schedule import options


def reporter():
    path = Path(__file__).resolve().parents[2] / "scripts/sparse/audit_hybrid_study.py"
    spec = importlib.util.spec_from_file_location("study_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def study(root):
    row = next(generate_candidates(HybridConfig.from_mapping(options()), (0,), (), (0,)))
    identity = row["candidate_id"]
    cells = request_cells([identity], ["input"], repeats=1, controls=("dense_strong",))
    references, journal = {}, []
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    for cell in cells:
        key = stable_hash([cell["variant"], cell["input_id"]])
        expected = root / (key + "-ref.npy")
        actual = root / (key + "-actual.npy")
        np.save(expected, np.zeros((32, 7), dtype=np.float32))
        np.save(actual, np.zeros((32, 7), dtype=np.float32))
        references[key] = dict(path=expected.name, sha256=sha(expected))
        journal.append(dict(**cell, seconds=0.1, own_eager_parity=True, attempt_id=0,
            actions_path=actual.name, actions_sha256=sha(actual), counters=dict(action_layer_updates=300),
            action_diagnostics=dict(relative_l2=0)))
    (root / "manifest.json").write_text(json.dumps(dict(status="COMPLETE", identity="fixture",
        identity_data=dict(git="test"), cells=cells, references=references,
        candidates={identity: row["options"]})))
    (root / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in journal))
    return journal


def test_study_audits_full_coverage_and_rejects_signed_zero_changes(tmp_path):
    rows = study(tmp_path)
    assert reporter().audit(tmp_path)["bitwise_eager_actions"] == 2
    path = tmp_path / rows[0]["actions_path"]
    changed = np.load(path, allow_pickle=False)
    changed[0, 0] = -0.0
    np.save(path, changed)
    rows[0]["actions_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="action bytes"):
        reporter().audit(tmp_path)


def test_study_does_not_summarize_incomplete_as_complete(tmp_path):
    rows = study(tmp_path)
    (tmp_path / "requests.jsonl").write_text(json.dumps(rows[0]) + "\n")
    with pytest.raises(ValueError, match="complete timing coverage"):
        reporter().audit(tmp_path)
