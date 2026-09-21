"""Adaptive decisions must precede execution, respect budgets and replay exactly."""

from dataclasses import replace

import pytest
import torch

from dreamwam.sparse.hybrid import HybridConfig
from dreamwam.sparse.hybrid.runtime import HybridVisualRuntime
from dreamwam.sparse.hybrid.schedule import Schedule
from dreamwam.sparse.hybrid.step_routing import StepRouterConfig
from dreamwam.sparse.hybrid.trace import HybridTrace
from test_hybrid_compact import compact
from test_visual_step_cache import model_and_inputs


def decision(config, **kwargs):
    values = dict(step=1, score=0, route_age=1, feature_age=1,
                  remaining=12, full_rows=12, sparse_rows=3)
    return config.decide(**{**values, **kwargs})


def test_causal_route_decisions_age_pressure_and_exact_budget_fallback():
    config = StepRouterConfig()
    assert decision(config, step=0)[0] == "dense"
    assert decision(config)[0] == "reuse"
    assert decision(config, score=.06)[0] == "sparse"
    assert decision(config, score=.4)[0] == "dense"
    assert decision(config, route_age=4)[0] == "sparse"
    assert decision(config, feature_age=8)[0] == "dense"
    assert decision(config, score=.4, remaining=3) == ("sparse", "dense_unaffordable_sparse")
    assert decision(config, score=.4, remaining=2) == ("reuse", "query_budget_exhausted")
    with pytest.raises(FloatingPointError): decision(config, score=float("nan"))


@pytest.mark.parametrize("patch", [dict(sparse_drift=-1), dict(dense_drift=.01),
    dict(extra_dense_budget=True), dict(max_reuse_steps=0), dict(max_feature_age=1.5)])
def test_router_configuration_rejects_invalid_thresholds(patch):
    with pytest.raises(ValueError): StepRouterConfig(**patch)


@pytest.mark.parametrize("backend", ["eager", "buffered", "cuda_graph"])
@pytest.mark.parametrize("dense_threshold", [0.000001, 1000])
def test_adaptive_execution_is_its_recorded_fixed_plan_and_clears_every_chunk(backend, dense_threshold):
    if backend == "cuda_graph" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    device = "cuda" if backend == "cuda_graph" else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    with torch.device(device):
        model, inputs = model_and_inputs()
    model.to(dtype=dtype)
    inputs = {key: value.to(device=device, dtype=dtype if value.is_floating_point() else value.dtype)
              if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}
    config = replace(compact(selection="action_context", backend=backend), step_router=StepRouterConfig(
        sparse_drift=0, dense_drift=dense_threshold, extra_dense_budget=1, max_reuse_steps=4))
    assert HybridConfig.from_mapping(config.describe()) == config
    with HybridVisualRuntime(model, config) as runtime:
        actual = model.sample_action(**inputs)
        stats = runtime.last_stats
        ops = tuple(row["effective_op"] for row in stats["steps"])
        assert ops[0] == "dense"
        assert stats["action_layer_updates"] == 8
        assert stats["query_rows_spent"] == sum(row["q_rows"] for row in stats["steps"])
        assert stats["query_rows_spent"] <= stats["query_row_cap"] == 24
        assert stats["computed_video_token_layers"] == stats["query_rows_spent"] * 2
        assert stats["step_router_probe_rows"] == 36
        assert stats["step_router_seconds"] > 0
        assert runtime.state.output is None
        assert torch.equal(actual, model.sample_action(**inputs))
    frozen = replace(config, step_router=None, backend="eager", schedule=Schedule(4, ops))
    with HybridVisualRuntime(model, frozen):
        assert torch.equal(actual, model.sample_action(**inputs))


def test_zero_extra_budget_never_probes_selection_or_refreshes_after_anchor():
    model, inputs = model_and_inputs()
    config = replace(compact(selection="action_context"), step_router=StepRouterConfig(extra_dense_budget=0))
    with HybridVisualRuntime(model, config) as runtime:
        with HybridTrace(runtime, layers=(0,)) as trace:
            model.sample_action(**inputs)
        assert [row["effective_op"] for row in runtime.last_stats["steps"]] == ["dense", "reuse", "reuse", "reuse"]
        assert runtime.last_stats["computed_video_token_layers"] == 24
        assert len([r for r in trace.records if r["kind"] == "router_probe"]) == 1


def test_seeded_random_is_equal_budget_and_does_not_touch_global_rng():
    model, inputs = model_and_inputs()
    routes = []
    for seed in (17, 18, 17):
        config = replace(compact(selection="random"), random_seed=seed)
        with HybridVisualRuntime(model, config) as runtime:
            # Sampling uses a private generator; selection must do the same.
            before = torch.random.get_rng_state().clone()
            model.sample_action(**inputs)
            assert torch.equal(before, torch.random.get_rng_state())
            rows = runtime.last_stats["steps"]
            assert [row["q_rows"] for row in rows] == [12, 3, 0, 3]
            assert all(len(row["route"]) == 6 for row in rows)
            routes.append([row["route"] for row in rows])
    assert routes[0] == routes[2] and routes[0] != routes[1]
