"""Real Joint sampler parity for graph input/output and cache handoff plumbing."""

import pytest
import torch

from dreamwam.sparse.action_guided_visual_token_cache import ActionGuidedVisualTokenCache
from dreamwam.sparse.visual_cache_graphs import GraphedVisualTokenCache, TensorTreeReplay
from test_visual_step_cache import model_and_inputs


@torch.no_grad()
def test_tree_buffers_refresh_all_inputs_and_do_not_alias_outputs():
    def compute(state, scale):
        return {"value": state["tokens"] * scale, "cache": [state["tokens"] + 1]}
    request = dict(state={"tokens": torch.ones(2, 3)}, scale=2)
    replay = TensorTreeReplay(compute, request, device="cpu", enabled=False)
    first = replay(request)
    request["state"]["tokens"].fill_(5)
    second = replay(request)
    second["cache"][0].fill_(100)
    assert torch.equal(first["value"], torch.full((2, 3), 2.0))
    assert torch.equal(replay(request)["cache"][0], torch.full((2, 3), 6.0))
    for changed in (dict(state={"tokens": torch.ones(2, 3)}, scale=3),
                    dict(state={"tokens": torch.ones(2, 3, dtype=torch.float64)}, scale=2),
                    dict(state={"tokens": torch.ones(3, 2)}, scale=2)):
        with pytest.raises(ValueError, match="graph input"):
            replay(changed)


@pytest.mark.parametrize("keep,interval", [(1.0, 1), (0.25, 2), (0.0, 2)])
@torch.no_grad()
def test_buffered_joint_matches_eager_across_changed_requests_and_reentries(keep, interval):
    model, inputs = model_and_inputs()
    variants = [inputs, {**inputs, "context": inputs["context"] * 0.4,
                         "proprio": inputs["proprio"] * -2,
                         "first_frame_latents": inputs["first_frame_latents"] + 0.3}, inputs]
    eager = ActionGuidedVisualTokenCache(model, keep_ratio=keep, refresh_every=interval)
    buffered = GraphedVisualTokenCache(model, keep_ratio=keep, refresh_every=interval,
                                      graph_enabled=False)
    original_action = model.mot.forward_action_with_video_cache
    original_joint = model.mot.forward
    for request in variants:
        with eager:
            expected = model.sample_action(**request)
            gates = {key: [value.clone() for value in values]
                     for key, values in model.world_residual._runtime_gates.items()}
        with buffered:
            actual = model.sample_action(**request)
            assert torch.equal(actual, expected)
            for key, values in gates.items():
                current = model.world_residual._runtime_gates[key]
                assert len(current) == len(values)
                assert all(torch.equal(left, right) for left, right in zip(current, values))
            for key, value in eager.last_stats.items():
                assert buffered.last_stats[key] == value
            assert not buffered.video_kv and buffered.video_output is None
        assert model.mot.forward_action_with_video_cache == original_action
        assert model.mot.forward == original_joint
    buffered.close_graphs()
    assert not buffered.graphs


def test_cuda_capture_cannot_silently_fall_back_to_cpu():
    with pytest.raises(ValueError, match="CUDA device"):
        TensorTreeReplay(lambda x: x, {"x": torch.ones(1)}, device="cpu")


@torch.no_grad()
def test_graph_failure_restores_request_and_action_method(monkeypatch):
    model, inputs = model_and_inputs()
    cache = GraphedVisualTokenCache(model, keep_ratio=0.25, refresh_every=2,
                                  graph_enabled=False)
    original = model.mot.forward_action_with_video_cache
    normal_call = cache._call
    def fail_action(name, *args, **kwargs):
        if name == "action":
            raise RuntimeError("capture interrupted")
        return normal_call(name, *args, **kwargs)
    monkeypatch.setattr(cache, "_call", fail_action)
    with cache:
        with pytest.raises(RuntimeError, match="capture interrupted"):
            model.sample_action(**inputs)
        assert not cache._active and not cache.video_kv
        assert model.mot.forward_action_with_video_cache == original
        monkeypatch.setattr(cache, "_call", normal_call)
        assert torch.isfinite(model.sample_action(**inputs)).all()
    cache.close_graphs()
