"""Verify physical query compression, complete keys, and cache lifetime."""

import pytest
import torch

import dreamwam.sparse.visual_token_cache as token_module
from dreamwam.sparse.visual_step_cache import VisualStepCache
from dreamwam.sparse.visual_token_cache import VisualTokenRefreshCache
from test_visual_step_cache import model_and_inputs


@pytest.mark.parametrize("interval", [1, 2])
def test_full_token_budget_matches_the_same_dense_refresh_schedule(interval):
    model, inputs = model_and_inputs()
    with VisualStepCache(model, refresh_every=interval):
        expected = model.sample_action(**inputs)
    with VisualTokenRefreshCache(model, keep_ratio=1, refresh_every=interval) as cache:
        assert torch.equal(model.sample_action(**inputs), expected)
        assert cache.last_stats["partial_video_steps"] == 0
        assert cache.last_stats["mask_builds"] == 0


def test_zero_token_budget_is_bitwise_one_anchor_temporal_reuse():
    model, inputs = model_and_inputs()
    with VisualStepCache(model, refresh_every=inputs["num_steps"]):
        expected = model.sample_action(**inputs)
    with VisualTokenRefreshCache(model, keep_ratio=0, refresh_every=2) as cache:
        assert torch.equal(model.sample_action(**inputs), expected)
        assert cache.last_stats["dense_video_steps"] == 1
        assert cache.last_stats["reused_video_steps"] == 3


def test_only_selected_visual_queries_execute_but_every_action_and_scheduler_does(monkeypatch):
    model, inputs = model_and_inputs()
    lengths = {"video": [], "action": []}
    handles = []
    for name, expert in (("video", model.video_expert), ("action", model.action_expert)):
        for block in expert.blocks:
            def hook(module, args, result, key=name):
                lengths[key].append(args[0].shape[1])
            handles.append(block.self_attn.q.register_forward_hook(hook))
    scheduler_calls = {"video": 0, "action": 0}
    for name, scheduler in (("video", model.video_scheduler), ("action", model.action_scheduler)):
        original = scheduler.step
        def step(*args, _original=original, _name=name, **kwargs):
            scheduler_calls[_name] += 1
            return _original(*args, **kwargs)
        monkeypatch.setattr(scheduler, "step", step)
    with VisualTokenRefreshCache(model, keep_ratio=0.25, refresh_every=2) as cache:
        actual = model.sample_action(**inputs)
        assert lengths["video"] == [12, 12, 3, 3]
        assert lengths["action"] == [4] * 8
        assert scheduler_calls == {"video": 4, "action": 4}
        assert cache.last_stats["computed_video_token_layers"] == 30
        assert cache.last_stats["total_video_token_layers"] == 96
        assert min(cache.last_stats["selected_indices"][0]) >= 4, "unchanged first-frame inputs have zero drift"
        assert cache.reference_input is None and not cache.video_kv
        assert torch.equal(model.sample_action(**inputs), actual)
    for handle in handles:
        handle.remove()


def test_partial_attention_keeps_all_keys_and_original_first_frame_visibility(monkeypatch):
    model, inputs = model_and_inputs()
    shapes = []
    original = token_module.scaled_dot_product_attention
    def checked(query, key, value, heads, mask):
        shapes.append((query.shape[1], key.shape[1], value.shape[1]))
        assert mask.shape == (7, 16)
        assert mask[0, :4].all() and not mask[0, 4:].any()
        assert mask[1:3, :12].all() and not mask[1:3, 12:].any()
        assert mask[3:].all()
        return original(query, key, value, heads, mask)
    monkeypatch.setattr(token_module, "scaled_dot_product_attention", checked)
    with VisualTokenRefreshCache(model, keep_ratio=0.25, refresh_every=2) as cache:
        monkeypatch.setattr(cache, "_select", lambda current, keep: torch.tensor([0, 5, 8], device=current.device))
        assert torch.isfinite(model.sample_action(**inputs)).all()
    assert shapes == [(7, 16, 16)] * 2


def test_unselected_kv_and_final_output_rows_are_retained(monkeypatch):
    model, inputs = model_and_inputs()
    cache = VisualTokenRefreshCache(model, keep_ratio=0.25, refresh_every=2)
    original = cache._partial_refresh
    checked = []
    def partial(video_state, action_state, residual, index, video):
        unselected = torch.ones(video_state["tokens"].shape[1], dtype=torch.bool)
        unselected[index.cpu()] = False
        before = {layer: {name: tensor[:, unselected].clone() for name, tensor in values.items()}
                  for layer, values in cache.video_kv.items()}
        result = original(video_state, action_state, residual, index, video)
        for layer, values in before.items():
            for name, tensor in values.items():
                assert torch.equal(cache.video_kv[layer][name][:, unselected], tensor)
        checked.append(True)
        return result
    monkeypatch.setattr(cache, "_partial_refresh", partial)
    original_refresh = cache._refresh
    def refresh(forward, **kwargs):
        before = None if cache.video_output is None else cache.video_output.clone()
        result = original_refresh(forward, **kwargs)
        if before is not None:
            unselected = torch.ones(before.shape[1], dtype=torch.bool)
            unselected[cache._indices[-1].cpu()] = False
            assert torch.equal(result["video"][:, unselected], before[:, unselected])
        return result
    monkeypatch.setattr(cache, "_refresh", refresh)
    with cache:
        model.sample_action(**inputs)
    assert checked == [True]


def test_partial_failure_discards_mutated_kv_before_the_next_request(monkeypatch):
    model, inputs = model_and_inputs()
    original = token_module.scaled_dot_product_attention
    def fail(*args, **kwargs):
        raise RuntimeError("partial attention interrupted")
    cache = VisualTokenRefreshCache(model, keep_ratio=0.25, refresh_every=2)
    with cache:
        monkeypatch.setattr(token_module, "scaled_dot_product_attention", fail)
        with pytest.raises(RuntimeError, match="partial attention"):
            model.sample_action(**inputs)
        assert not cache.video_kv and cache.video_output is None
        assert cache.reference_input is None and not cache._active
        monkeypatch.setattr(token_module, "scaled_dot_product_attention", original)
        actual = model.sample_action(**inputs)
        assert torch.equal(model.sample_action(**inputs), actual)


def test_shared_mask_handles_a_batch_without_changing_action_batch_size():
    model, inputs = model_and_inputs()
    for name in ("first_frame_latents", "context", "context_mask", "proprio"):
        value = inputs[name]
        inputs[name] = value.repeat(2, *([1] * (value.ndim - 1)))
    with VisualTokenRefreshCache(model, keep_ratio=0.25, refresh_every=2):
        actual = model.sample_action(**inputs)
        assert actual.shape[:2] == (2, 4)
        assert torch.isfinite(actual).all()
