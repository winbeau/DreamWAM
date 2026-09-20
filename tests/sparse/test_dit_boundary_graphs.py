import pytest
import torch

from dreamwam.sparse.dense_graph_control import NativeDenseGraph
from dreamwam.sparse.dit_boundary_graphs import BoundaryGraphControl, DiTBoundaryGraphs
from dreamwam.sparse.visual_cache_graphs import GraphedVisualTokenCache
from dreamwam.sparse.visual_step_cache import VisualStepCache
from test_visual_step_cache import model_and_inputs


@pytest.mark.parametrize("temporal", [False, True])
@pytest.mark.parametrize("cuda", [False, True])
@torch.no_grad()
def test_boundaries_preserve_joint_actions_world_diagnostics_and_every_step(temporal, cuda):
    if cuda and not torch.cuda.is_available():
        pytest.skip("CUDA graph integration")
    with torch.device("cuda" if cuda else "cpu"):
        model, inputs = model_and_inputs()
    reference = VisualStepCache(model, refresh_every=2 if temporal else 1)
    transformer = (GraphedVisualTokenCache(model, keep_ratio=1.0, refresh_every=2,
                                           graph_enabled=cuda, graph_partial=True)
                   if temporal else NativeDenseGraph(model, graph_enabled=cuda))
    control = BoundaryGraphControl(transformer, enabled=cuda)
    original = (model.sample_action, model.video_expert.pre_dit, model.video_expert.post_dit,
                model.action_expert.pre_dit, model.action_expert.post_dit)
    for request in (inputs, {**inputs, "context": inputs["context"] * 0.3,
                            "proprio": inputs["proprio"] * -2,
                            "first_frame_latents": inputs["first_frame_latents"] + 0.2}, inputs):
        with reference:
            expected = model.sample_action(**request)
            gates = {key: [value.clone() for value in values]
                     for key, values in model.world_residual._runtime_gates.items()}
        with control:
            actual = model.sample_action(**request)
            assert torch.equal(actual, expected)
            for name in DiTBoundaryGraphs.names:
                assert control.last_stats[f"dit_{name}_calls"] == 4
                assert control.last_stats[f"dit_{name}_graph_replays"] == (4 if cuda else 0)
            for key, values in gates.items():
                current = model.world_residual._runtime_gates[key]
                assert len(current) == len(values)
                assert all(torch.equal(left, right) for left, right in zip(values, current))
        assert original == (model.sample_action, model.video_expert.pre_dit, model.video_expert.post_dit,
                            model.action_expert.pre_dit, model.action_expert.post_dit)
        assert not getattr(model, "_dit_boundary_graphs", None)
    control.close_graphs()
    assert not control.boundaries.graphs


@torch.no_grad()
def test_boundary_failure_restores_methods_and_request_state():
    model, inputs = model_and_inputs()
    original = model.sample_action
    control = BoundaryGraphControl(NativeDenseGraph(model, graph_enabled=False), enabled=False)
    with pytest.raises(ValueError):
        with control:
            model.sample_action(**{**inputs, "num_steps": 0})
    assert model.sample_action == original
    assert not control.boundaries._active and not control.transformer._active
    assert not getattr(model, "_dit_boundary_graphs", None)
    with control:
        assert torch.isfinite(model.sample_action(**inputs)).all()
    control.close_graphs()
