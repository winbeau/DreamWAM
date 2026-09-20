import pytest
import torch

from dreamwam.sparse.dense_graph_control import NativeDenseGraph
from test_visual_step_cache import model_and_inputs


@torch.no_grad()
def test_native_dense_buffering_preserves_actions_diagnostics_and_request_boundaries():
    model, inputs = model_and_inputs()
    graph = NativeDenseGraph(model, graph_enabled=False)
    original_sample, original_forward = model.sample_action, model.mot.forward
    for request in (inputs, {**inputs, "proprio": inputs["proprio"] * -2,
                            "first_frame_latents": inputs["first_frame_latents"] + 0.5}, inputs):
        expected = model.sample_action(**request)
        gates = {name: [value.clone() for value in values]
                 for name, values in model.world_residual._runtime_gates.items()}
        with graph:
            assert torch.equal(model.sample_action(**request), expected)
            assert graph.last_stats == dict(dense_transformer_calls=4, dense_graph_replays=0,
                                            exported_visual_kv_layers=0)
            for name, values in gates.items():
                current = model.world_residual._runtime_gates[name]
                assert len(current) == len(values)
                assert all(torch.equal(left, right) for left, right in zip(values, current))
        assert model.sample_action == original_sample and model.mot.forward == original_forward
    graph.close_graphs()
    assert graph.replay is None


@torch.no_grad()
def test_failed_native_graph_call_clears_active_request(monkeypatch):
    model, inputs = model_and_inputs()
    with NativeDenseGraph(model, graph_enabled=False) as graph:
        with pytest.raises(ValueError):
            model.sample_action(**{**inputs, "num_steps": 0})
        assert not graph._active
        assert torch.isfinite(model.sample_action(**inputs)).all()
    graph.close_graphs()
