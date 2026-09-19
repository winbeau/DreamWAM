"""Action relevance changes selection without changing action-step execution."""

import pytest
import torch

import dreamwam.sparse.action_guided_visual_token_cache as guided_module
from dreamwam.sparse.action_guided_visual_token_cache import ActionGuidedVisualTokenCache
from dreamwam.sparse.visual_token_cache import VisualTokenRefreshCache
from test_visual_step_cache import model_and_inputs


def test_zero_guidance_is_bitwise_the_unguided_factor():
    model, inputs = model_and_inputs()
    with VisualTokenRefreshCache(model, keep_ratio=0.25, refresh_every=2):
        expected = model.sample_action(**inputs)
    with ActionGuidedVisualTokenCache(model, keep_ratio=0.25, refresh_every=2, guidance_weight=0) as cache:
        assert torch.equal(model.sample_action(**inputs), expected)
        assert cache.last_stats["action_mass_builds"] == 0


def test_full_budget_is_bitwise_native_dense_without_guidance_overhead():
    model, inputs = model_and_inputs()
    expected = model.sample_action(**inputs)
    with ActionGuidedVisualTokenCache(model, keep_ratio=1, refresh_every=1) as cache:
        assert torch.equal(model.sample_action(**inputs), expected)
        assert cache.last_stats["guidance_action_qkv_builds"] == 0


def test_current_action_read_is_amortized_over_layers_and_does_not_add_denoising_steps(monkeypatch):
    model, inputs = model_and_inputs()
    calls = dict(action_qkv=0, action_ffn=0, action_scheduler=0)
    handles = []
    def count(name):
        def hook(module, args, result):
            calls[name] += 1
        return hook
    for block in model.action_expert.blocks:
        handles.append(block.self_attn.q.register_forward_hook(count("action_qkv")))
        handles.append(block.ffn.register_forward_hook(count("action_ffn")))
    original = model.action_scheduler.step
    def step(*args, **kwargs):
        calls["action_scheduler"] += 1
        return original(*args, **kwargs)
    monkeypatch.setattr(model.action_scheduler, "step", step)
    with ActionGuidedVisualTokenCache(model, keep_ratio=0.25, refresh_every=2) as cache:
        actual = model.sample_action(**inputs)
        assert calls == dict(action_qkv=9, action_ffn=8, action_scheduler=4)
        assert cache.last_stats["action_mass_builds"] == 1
        assert cache.last_stats["guidance_action_qkv_builds"] == 1
        assert cache.last_stats["action_layer_updates"] == 8
        assert min(cache.last_stats["selected_indices"][0]) >= 4
        assert cache.action_mass is None and cache.reference_input is None
        assert torch.equal(model.sample_action(**inputs), actual)
    for handle in handles:
        handle.remove()


def test_relevance_can_change_selection_but_cannot_promote_zero_drift():
    model, _ = model_and_inputs()
    current = torch.ones(1, 3, 2)
    reference = torch.tensor([[[0.4, 0.4], [0.5, 0.5], [1.0, 1.0]]])
    plain = VisualTokenRefreshCache(model, keep_ratio=1 / 3)
    plain.reference_input = reference
    assert plain._select(current, 1).tolist() == [0]
    guided = ActionGuidedVisualTokenCache(model, keep_ratio=1 / 3)
    guided.reference_input = reference
    guided.action_mass = torch.tensor([[0.0, 1.0, 5.0]])
    assert guided._select(current, 1).tolist() == [1]


def test_failed_mass_extraction_does_not_leak_request_state(monkeypatch):
    model, inputs = model_and_inputs()
    original = guided_module.joint_video_mass
    def fail(*args, **kwargs):
        raise RuntimeError("mass extraction interrupted")
    with ActionGuidedVisualTokenCache(model, keep_ratio=0.25, refresh_every=2) as cache:
        monkeypatch.setattr(guided_module, "joint_video_mass", fail)
        with pytest.raises(RuntimeError, match="mass extraction"):
            model.sample_action(**inputs)
        assert cache.action_mass is None and not cache.video_kv and not cache._active
        monkeypatch.setattr(guided_module, "joint_video_mass", original)
        assert torch.isfinite(model.sample_action(**inputs)).all()
