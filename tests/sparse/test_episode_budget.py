from concurrent.futures import ThreadPoolExecutor
import json

import pytest

from dreamwam.sparse.episode_budget import EpisodeBudget


def test_failed_or_incomplete_runs_do_not_refund_the_bounded_budget(tmp_path):
    path = tmp_path / "ledger.json"
    ledger = EpisodeBudget(path)
    assert ledger.available() == 50
    receipt = ledger.reserve("pair-one", 6, metadata={"planned_arms": 2})
    assert receipt["total_charged"] == 6 and ledger.available() == 44
    finished = ledger.finish("pair-one", evidence={"recorded_attempts": 1, "exact_attempt_count": False})
    assert finished["charged"] == 6 and ledger.available() == 44
    assert EpisodeBudget(path).available() == 44
    with pytest.raises(ValueError, match="already exists"):
        ledger.reserve("pair-one", 6, metadata={})
    with pytest.raises(ValueError, match="already finalized"):
        ledger.finish("pair-one", evidence={})
    with pytest.raises(ValueError, match="budget exhausted"):
        ledger.reserve("pair-too-large", 45, metadata={})
    assert len(json.loads(path.read_text())["reservations"]) == 1


def test_concurrent_pair_reservations_cannot_oversubscribe(tmp_path):
    path = tmp_path / "ledger.json"
    def reserve(name):
        try:
            return EpisodeBudget(path, cap=10).reserve(name, 6, metadata={})
        except ValueError as exc:
            return str(exc)
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(reserve, ("a", "b")))
    assert sum(isinstance(result, dict) for result in results) == 1
    assert sum(isinstance(result, str) and "budget exhausted" in result for result in results) == 1
    assert EpisodeBudget(path, cap=10).available() == 4
    with pytest.raises(ValueError, match="identity/cap"):
        EpisodeBudget(path, cap=50).available()


def test_corrupt_or_changed_ledgers_fail_closed(tmp_path):
    path = tmp_path / "ledger.json"
    ledger = EpisodeBudget(path)
    ledger.reserve("existing", 3, metadata={})
    before = path.read_bytes()
    with pytest.raises(ValueError, match="identity/cap"):
        EpisodeBudget(path, effort="another-project").reserve("x", 3, metadata={})
    assert path.read_bytes() == before
    path.write_text("not json")
    with pytest.raises(json.JSONDecodeError):
        ledger.reserve("x", 3, metadata={})
    assert path.read_text() == "not json"
    with pytest.raises(ValueError, match="existing conserved ledger"):
        EpisodeBudget(tmp_path / "too-large.json", cap=51).available()


def test_explicit_extension_preserves_history_and_bounds_concurrent_new_work(tmp_path):
    import hashlib
    path = tmp_path / "ledger.json"
    ledger = EpisodeBudget(path)
    ledger.reserve("historical", 32, metadata={"frozen": True})
    ledger.finish("historical", evidence={"recorded_attempts": 32})
    before = path.read_bytes()
    amendment = ledger.amend_cap(132, authorization="50 pairs, 100 new episodes", expected_charged=32)
    after = json.loads(path.read_text())
    assert after["reservations"] == json.loads(before)["reservations"]
    assert amendment["previous_ledger_sha256"] == hashlib.sha256(before).hexdigest()
    assert EpisodeBudget(path, cap=132).available() == 100
    def reserve(name):
        return EpisodeBudget(path, cap=132).reserve(name, 50, metadata={})
    with ThreadPoolExecutor(max_workers=2) as executor:
        assert len(list(executor.map(reserve, ("lane-a", "lane-b")))) == 2
    assert EpisodeBudget(path, cap=132).available() == 0
    with pytest.raises(ValueError, match="exhausted"):
        EpisodeBudget(path, cap=132).reserve("extra", 1, metadata={})
    with pytest.raises(ValueError, match="identity/cap"):
        EpisodeBudget(path).available()


def test_extension_rejects_pending_stale_or_unrecorded_scope(tmp_path):
    path = tmp_path / "ledger.json"
    ledger = EpisodeBudget(path)
    ledger.reserve("active", 32, metadata={})
    before = path.read_bytes()
    with pytest.raises(ValueError, match="finalized history"):
        ledger.amend_cap(132, authorization="100 new", expected_charged=32)
    assert path.read_bytes() == before
    ledger.finish("active", evidence={})
    before = path.read_bytes()
    for authorization, expected in (("", 32), ("100 new", 31)):
        with pytest.raises(ValueError):
            ledger.amend_cap(132, authorization=authorization, expected_charged=expected)
        assert path.read_bytes() == before
    corrupt = json.loads(before)
    corrupt["cap"] = 132
    path.write_text(json.dumps(corrupt))
    with pytest.raises(ValueError, match="recorded authorization"):
        EpisodeBudget(path, cap=132).available()
