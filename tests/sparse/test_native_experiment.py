from dreamwam.sparse.hybrid.config import HybridConfig
from dreamwam.sparse.hybrid.experiment import request_cells
from dreamwam.sparse.hybrid.native_experiment import native_candidates, planned_calls, uniform_feature_control


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
