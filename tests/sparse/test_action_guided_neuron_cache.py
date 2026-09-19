"""Check causal action guidance without changing attention or cache lifetime."""

import pytest
import torch

from dreamwam.sparse.action_guided_neuron_cache import (
    ActionGuidedFFNNeuronCache,
    joint_video_mass,
)
from dreamwam.sparse.ffn_neuron_cache import VisualFFNNeuronCache
from test_visual_step_cache import model_and_inputs


def test_action_keys_compete_with_video_keys_and_visibility_is_respected():
    query = torch.ones(1, 2, 2)
    video_key = torch.zeros(1, 3, 2)
    action_key = torch.zeros(1, 2, 2)
    mask = torch.ones(2, 5, dtype=torch.bool)
    mass = joint_video_mass(query, video_key, action_key, num_heads=1, action_mask=mask)
    torch.testing.assert_close(mass, torch.full((1, 3), 2 / 5))
    competing = joint_video_mass(query, video_key, action_key + 10, num_heads=1, action_mask=mask)
    assert torch.all(competing < mass / 100)
    mask[:, 1] = False
    masked = joint_video_mass(query, video_key, action_key, num_heads=1, action_mask=mask)
    assert masked[0, 1] == 0


def test_zero_guidance_is_bitwise_unguided_and_full_budget_is_bitwise_dense():
    model, inputs = model_and_inputs()
    dense = model.sample_action(**inputs)
    plain = VisualFFNNeuronCache(model, keep_ratio=0.25, group_size=2)
    plain.prepare()
    with plain:
        unguided = model.sample_action(**inputs)
    guided = ActionGuidedFFNNeuronCache(model, keep_ratio=0.25, group_size=2, guidance_weight=0)
    guided.prepare()
    with guided:
        assert torch.equal(model.sample_action(**inputs), unguided)
        assert guided.last_stats["action_mass_builds"] == 0
    with ActionGuidedFFNNeuronCache(model, keep_ratio=1.0) as full:
        assert torch.equal(model.sample_action(**inputs), dense)
        assert full.last_stats["action_mass_builds"] == 0


def test_mass_is_only_built_at_group_anchors_and_requests_are_isolated():
    model, inputs = model_and_inputs()
    original = model.mot._joint_self_attention
    cache = ActionGuidedFFNNeuronCache(model, keep_ratio=0.25, group_size=2)
    cache.prepare()
    with cache:
        actual = model.sample_action(**inputs)
        assert torch.isfinite(actual).all()
        assert cache.last_stats["action_mass_builds"] == 4  # 2 layers x 2 anchors
        assert cache.last_stats["guided_mask_builds"] == 4
        assert cache.last_stats["mask_builds"] == 4
        assert not cache.action_mass and not cache.references
        assert torch.equal(model.sample_action(**inputs), actual)
    assert model.mot._joint_self_attention == original


def test_action_relevance_can_change_which_neuron_is_selected():
    model, _ = model_and_inputs()
    cache = ActionGuidedFFNNeuronCache(model, keep_ratio=0.5, guidance_weight=1)
    cache.weight_norms[0] = torch.ones(2)
    hidden = torch.tensor([[[0.0, 4.0], [5.0, 0.0]]])
    assert hidden.abs().sum((0, 1)).argmax() == 0
    cache._begin()
    cache.action_mass[0] = torch.tensor([[1.0, 0.0]])
    assert cache._neuron_score(0, hidden).argmax() == 1
    cache._end()


def test_exception_after_mass_capture_clears_request_tensors(monkeypatch):
    model, inputs = model_and_inputs()
    def fail(**kwargs):
        raise RuntimeError("attention failed after capture")
    monkeypatch.setattr(model.mot, "_joint_self_attention", fail)
    cache = ActionGuidedFFNNeuronCache(model, keep_ratio=0.1)
    cache.prepare()
    with cache:
        with pytest.raises(RuntimeError, match="after capture"):
            model.sample_action(**inputs)
        assert not cache.action_mass and not cache.references and not cache._active
