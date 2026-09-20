"""Structure reuse is not feature reuse, including untouched rows and graph IO."""

from dataclasses import replace

import pytest
import torch

from dreamwam.sparse.hybrid.runtime import HybridVisualRuntime
from test_hybrid_compact import compact
from test_visual_step_cache import model_and_inputs


def structure(**kwargs):
    return replace(compact(selection="action_context", **kwargs),
                   recompute_ratio=0.5, reuse_mode="structure")


def test_structure_reuse_recomputes_current_qkv_without_reading_old_features(monkeypatch):
    model, inputs = model_and_inputs()
    conf = structure()
    with HybridVisualRuntime(model, conf) as runtime:
        expected = model.sample_action(**inputs)
        original = runtime._call
        fresh_calls = []
        def poison(operation, request):
            if operation == "fresh":
                assert set(request) == {"video_state", "action_state", "mask"}
                for layer in runtime.state.kv:
                    for tensor in layer.values():
                        tensor.fill_(float("nan"))
                runtime.state.output.fill_(float("nan"))
                fresh_calls.append(True)
            return original(operation, request)
        monkeypatch.setattr(runtime, "_call", poison)
        actual = model.sample_action(**inputs)
        assert torch.equal(actual, expected)
        assert len(fresh_calls) == 3
        stats = runtime.last_stats
        assert stats["computed_video_token_layers"] == (12 + 6 * 3) * 2
        assert [row["q_rows"] for row in stats["steps"]] == [12, 6, 6, 6]
        assert [row["route_age"] for row in stats["steps"]] == [0, 0, 1, 0]
        assert not any(row["reused_visual_features"] for row in stats["steps"])


def test_unselected_rows_bypass_current_inputs_not_cached_outputs(monkeypatch):
    model, inputs = model_and_inputs()
    with HybridVisualRuntime(model, structure()) as runtime:
        commit = runtime.state.commit_fresh
        def check(result, current, route, step, **kwargs):
            untouched = torch.ones(current.shape[1], dtype=torch.bool)
            untouched[route] = False
            commit(result, current, route, step, **kwargs)
            assert torch.equal(runtime.state.output[:, untouched], current[:, untouched])
            assert not runtime.state.kv and not runtime.state.packed
        monkeypatch.setattr(runtime.state, "commit_fresh", check)
        model.sample_action(**inputs)


@pytest.mark.parametrize("backend", ["buffered", "cuda_graph"])
def test_structure_graphs_match_eager_with_changed_inputs(backend):
    if backend == "cuda_graph" and not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    device = "cuda" if backend == "cuda_graph" else "cpu"
    with torch.device(device):
        model, inputs = model_and_inputs()
    conf = structure()
    runtime = HybridVisualRuntime(model, replace(conf, backend=backend))
    for request in (inputs, {**inputs, "context": inputs["context"] + 2}, inputs):
        with HybridVisualRuntime(model, conf):
            expected = model.sample_action(**request)
        with runtime:
            assert torch.equal(model.sample_action(**request), expected)
    runtime.close_graphs()


def test_structure_full_budget_is_native_and_incompatible_options_fail():
    model, inputs = model_and_inputs()
    expected = model.sample_action(**inputs)
    with HybridVisualRuntime(model, replace(structure(), read_ratio=1, recompute_ratio=1)):
        assert torch.equal(model.sample_action(**inputs), expected)
    for changes in (dict(recompute_ratio=0.25), dict(selection="action_drift"), dict(read_mode="full", read_ratio=1)):
        with pytest.raises(ValueError):
            replace(structure(), **changes)
