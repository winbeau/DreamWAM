from dreamwam.sparse.hybrid.config import HybridConfig
from dreamwam.sparse.hybrid.experiment import request_cells
from dreamwam.sparse.hybrid.native_experiment import frozen_candidates, native_candidates, planned_calls, uniform_feature_control


def test_predeclared_stages_and_inclusive_call_cap():
    expected = {"selectors": (11, 219), "layers": (4, 76), "pooling": (5, 105)}
    for stage, (count, calls) in expected.items():
        candidates = native_candidates(stage)
        assert len(candidates) == count
        for _, config in candidates:
            assert HybridConfig.from_mapping(config.describe()).policy_hash == config.policy_hash
            assert config.budgets(294, 98) == (30, 198 if stage == "pooling" else 56)
            assert config.schedule.operations == ("dense",) + ("reuse",) * 9
        cells = request_cells([config.policy_hash for _, config in candidates], ["a", "b"],
            repeats=2, group_size=2, controls=("dense_strong", "uniform_features"))
        budget = planned_calls(cells, 2)
        assert budget["total"] == calls == sum(value for key, value in budget.items() if key != "total")
    control = uniform_feature_control()
    assert control.budgets(294, 98) == (30, 56)
    assert control.native_routing is None and control.selection == "drift"
    assert control.read_ratio == 0.1875  # exact historical option; effective count is 56


def test_pool_hard_controls_match_read_quotas_and_fusion_weights_are_explicit():
    pairs = dict(native_candidates("pooling"))
    for signal in ("uniform", "dynamic"):
        hard, pool = pairs["hard198_" + signal], pairs["pool198_" + signal]
        assert hard.read_count == pool.read_count == 198
        assert hard.native_routing.observed == pool.native_routing.observed == "full"
        assert hard.native_routing.signal == pool.native_routing.signal == signal
    mixtures = [config.native_routing for _, config in native_candidates("selectors")
                if config.native_routing.signal == "fusion"]
    assert {dict(native.weights)["dynamic"] for native in mixtures} == {0.25, 0.5, 0.75}
    assert all(sum(dict(native.weights).values()) == 1 for native in mixtures)


def test_refresh_scan_covers_every_single_position_and_structure_has_fresh_queries():
    refresh = dict(native_candidates("refresh"))
    assert len(refresh) == 10
    assert [config.schedule.operations.count("sparse") for config in refresh.values()] == [0] + [1] * 9
    for index in range(1, 10):
        config = refresh[f"context_sparse_{index}"]
        assert config.schedule.operations[index] == "sparse"
        assert config.budgets(294, 98) == (30, 56)
        assert config.native_routing.signal == "action_context" and config.native_routing.recompute == "drift"
    for _, config in native_candidates("structure"):
        assert config.reuse_mode == "structure" and config.budgets(294, 98) == (56, 56)
        assert "sparse" not in config.schedule.operations
    for stage, count in (("refresh", 190), ("structure", 38)):
        configs = [config for _, config in native_candidates(stage)]
        assert all(HybridConfig.from_mapping(c.describe()).policy_hash == c.policy_hash for c in configs)
        cells = request_cells([c.policy_hash for c in configs], ["a", "b"], repeats=2,
                              group_size=2, controls=("dense_strong", "uniform_features"))
        assert planned_calls(cells, 2)["total"] == count


def test_committed_frozen_matrices_fit_inclusive_budgets_and_preserve_read_query_separation():
    import json
    from pathlib import Path
    import pytest
    plan = json.loads((Path(__file__).resolve().parents[2] /
        "docs/implementation/dido-sparse-profile/experiment-plan.json").read_text())
    for design in (value for value in plan.values() if isinstance(value, dict) and isinstance(value.get("candidates"), list)):
        configs = frozen_candidates(design)
        cells = request_cells([c.policy_hash for _, c in configs], design["input_ids"], repeats=2,
                              group_size=2, controls=("dense_strong", "uniform_features"))
        assert planned_calls(cells, len(design["input_ids"]))["total"] == design["predict_call_cap"]
        with pytest.raises(ValueError, match="duplicate"):
            frozen_candidates({"candidates": [design["candidates"][0]] * 2})
    rows = dict(frozen_candidates(plan["online_budget_followup"]))
    for count in (28, 56, 84):
        assert rows[f"context_S9_R{count}_U15"].budgets(294, 98) == (15, count)
    assert rows["context_S9_R56_U45"].budgets(294, 98) == (45, 56)
    assert rows["context_S4_S9"].schedule.operations.count("sparse") == 2
    assert rows["context_S1_S4_S9"].schedule.operations.count("sparse") == 3
