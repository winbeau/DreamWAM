import pytest
import torch

from dreamwam.sparse.conditioned_frame_cache import ConditionedFrameCache, GraphedConditionedFrameCache
from dreamwam.sparse.visual_step_cache import VisualStepCache
from test_visual_step_cache import model_and_inputs


@pytest.mark.parametrize("interval", [1, 2])
@torch.no_grad()
def test_conditioned_reuse_matches_current_cadence_and_resets_changed_requests(interval):
    torch.manual_seed(19)
    model, inputs = model_and_inputs()
    original_forward = model.mot.forward
    cache = ConditionedFrameCache(model, refresh_every=interval)
    for request in (inputs, {**inputs, "first_frame_latents": inputs["first_frame_latents"] + 0.6,
                            "context": inputs["context"] * -1.2,
                            "proprio": inputs["proprio"] * -2}, inputs):
        with VisualStepCache(model, refresh_every=interval):
            expected = model.sample_action(**request)
            gates = {key: [value.clone() for value in values]
                     for key, values in model.world_residual._runtime_gates.items()}
        with cache:
            actual = model.sample_action(**request)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
            for key, values in gates.items():
                current = model.world_residual._runtime_gates[key]
                assert len(current) == len(values)
                for left, right in zip(values, current):
                    torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)
            refreshes = (request["num_steps"] - 1) // interval + 1
            assert cache.last_stats["conditioned_token_layer_reuses"] == 4 * 2 * (refreshes - 1)
            assert cache.last_stats["selected_indices"] == [list(range(4, 12))] * (refreshes - 1)
            assert cache.last_stats["action_layer_updates"] == 8
            assert not cache.video_kv and not cache._conditioned_gates
        assert model.mot.forward == original_forward


@torch.no_grad()
def test_buffered_conditioned_reuse_is_bitwise_its_eager_path_and_cleans_failure():
    model, inputs = model_and_inputs()
    eager = ConditionedFrameCache(model)
    buffered = GraphedConditionedFrameCache(model, graph_enabled=False)
    for request in (inputs, {**inputs, "proprio": inputs["proprio"] + 0.7}, inputs):
        with eager:
            expected = model.sample_action(**request)
        with buffered:
            assert torch.equal(model.sample_action(**request), expected)
            assert buffered.last_stats["dense_buffered_calls"] == 1
            assert buffered.last_stats["partial_buffered_calls"] == 3
    with buffered:
        with pytest.raises(ValueError):
            model.sample_action(**{**inputs, "num_steps": 0})
        assert not buffered._active and not buffered.video_kv and not buffered._conditioned_gates
        assert torch.isfinite(model.sample_action(**inputs)).all()
    buffered.close_graphs()
    assert not buffered.graphs
