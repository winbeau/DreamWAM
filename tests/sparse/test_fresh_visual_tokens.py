"""Check every-step physical sparsity, current inputs, visibility, and graph parity."""

import pytest
import torch
from torch.utils._pytree import tree_map

import dreamwam.sparse.fresh_visual_tokens as module
from dreamwam.sparse.fresh_visual_tokens import FreshVisualTokenSparsity, fresh_visual_options
from test_visual_step_cache import model_and_inputs


@pytest.mark.parametrize("buffered", [False, True])
def test_full_budget_is_bitwise_native_at_every_step(buffered):
    model, inputs = model_and_inputs()
    expected = model.sample_action(**inputs)
    with FreshVisualTokenSparsity(model, keep_ratio=1, buffered=buffered) as runtime:
        assert torch.equal(model.sample_action(**inputs), expected)
        assert runtime.last_stats["computed_video_token_layers"] == 96
        assert runtime.last_stats["reused_visual_steps"] == 0
    assert torch.equal(model.sample_action(**inputs), expected)


def test_every_step_has_compact_qkv_ffn_and_complete_actions(monkeypatch):
    model, inputs = model_and_inputs()
    observed = {k: [] for k in ("vq", "vk", "ffn", "aq")}
    handles = []
    for layer in range(model.mot.num_layers):
        vb, ab = model.video_expert.blocks[layer], model.action_expert.blocks[layer]
        for name, block in (("vq", vb.self_attn.q), ("vk", vb.self_attn.k),
                            ("ffn", vb.ffn), ("aq", ab.self_attn.q)):
            handles.append(block.register_forward_pre_hook(
                lambda _, args, key=name: observed[key].append(args[0].shape[1])))
    scheduled = []
    for name, scheduler in (("video", model.video_scheduler), ("action", model.action_scheduler)):
        original = scheduler.step
        def step(*args, _original=original, _name=name, **kwargs):
            scheduled.append(_name)
            return _original(*args, **kwargs)
        monkeypatch.setattr(scheduler, "step", step)
    with FreshVisualTokenSparsity(model, keep_ratio=0.25) as runtime:
        assert torch.isfinite(model.sample_action(**inputs)).all()
        assert observed["vq"] == observed["ffn"] == [3] * 8
        assert observed["vk"] == [12, 3, 3] * 4  # current probe, then two compact layers
        assert observed["aq"] == [4] * 8  # first action QKV is shared WITHIN this step
        assert scheduled == ["video", "action"] * 4
        assert runtime.last_stats["computed_video_token_layers"] == 24
        assert runtime.last_stats["selection_video_key_rows"] == 48
        assert runtime.last_stats["selection_builds"] == 4
        assert runtime.last_stats["reused_visual_steps"] == 0
        for index in runtime.last_stats["selected_indices"]:
            assert sorted(x // 4 for x in index) == [0, 1, 2]
    for handle in handles:
        handle.remove()


def test_attention_drops_inactive_keys_and_preserves_original_positions(monkeypatch):
    model, inputs = model_and_inputs()
    original = module.scaled_dot_product_attention
    calls = []
    def checked(q, k, v, heads, mask):
        assert q.shape[1] == k.shape[1] == v.shape[1] == 7
        assert mask.shape == (7, 7)
        assert mask[0].tolist() == [True, False, False, False, False, False, False]
        assert mask[1:3, :3].all() and not mask[:3, 3:].any()
        assert mask[3:].all()
        calls.append(True)
        return original(q, k, v, heads, mask)
    monkeypatch.setattr(module, "scaled_dot_product_attention", checked)
    with FreshVisualTokenSparsity(model, keep_ratio=0.25, selection="uniform"):
        model.sample_action(**inputs)
    assert len(calls) == 8


@torch.no_grad()
def test_compact_matches_independent_full_shape_masked_reference(monkeypatch):
    model, inputs = model_and_inputs()
    states = []
    handle = model.mot.register_forward_pre_hook(
        lambda _, args, kwargs: states.append(tree_map(
            lambda x: x.clone() if isinstance(x, torch.Tensor) else x,
            {k: kwargs[k] for k in ("video_state", "action_state")})), with_kwargs=True)
    model.sample_action(**inputs)
    handle.remove()
    vs, ac = states[0]["video_state"], states[0]["action_state"]
    index = torch.tensor([0, 5, 8])
    with FreshVisualTokenSparsity(model, keep_ratio=0.25, selection="uniform") as runtime:
        monkeypatch.setattr(runtime, "_select", lambda *args: (index, None))
        actual = runtime._compute(vs, ac)["result"]
        # Oracle computes ALL query/projected rows but masks inactive key columns.
        # Only the selected residual rows are updated after each full-shape layer.
        mot = model.mot
        video, action = vs["tokens"].clone(), ac["tokens"].clone()
        mask = mot.build_attention_mask(video_length=12, action_length=4,
                                       video_tokens_per_frame=4, device=video.device)
        keep_keys = torch.zeros(16, dtype=torch.bool)
        keep_keys[index] = True
        keep_keys[12:] = True
        mask = mask & keep_keys.unsqueeze(0)
        for layer in range(mot.num_layers):
            vb, ab = mot.video_expert.blocks[layer], mot.action_expert.blocks[layer]
            vio = mot._attention_input(vb, video, vs["freqs"], vs["time_modulation"])
            aio = mot._attention_input(ab, action, ac["freqs"], ac["time_modulation"])
            mixed = module.scaled_dot_product_attention(
                torch.cat((vio[0], aio[0]), 1), torch.cat((vio[1], aio[1]), 1),
                torch.cat((vio[2], aio[2]), 1), mot.num_heads, mask)
            full = mot._post_attention(vb, vio[3], mixed[:, :12], *vio[4:], vs["context"], vs["context_mask"])
            full = model.world_residual(layer, full, vs["context"], vs["context_mask"])
            video = video.index_copy(1, index, full.index_select(1, index))
            action = mot._post_attention(ab, aio[3], mixed[:, 12:], *aio[4:], ac["context"], ac["context_mask"])
        torch.testing.assert_close(actual["video"], video, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(actual["action"], action, rtol=2e-5, atol=2e-6)
        inactive = torch.ones(12, dtype=torch.bool)
        inactive[index] = False
        assert torch.equal(actual["video"][:, inactive], vs["tokens"][:, inactive])


@pytest.mark.parametrize("selection", ["action", "uniform"])
def test_buffered_requests_do_not_retain_previous_hidden_states(selection):
    model, inputs = model_and_inputs()
    requests = [inputs, {**inputs, "first_frame_latents": inputs["first_frame_latents"] + 2,
                        "proprio": inputs["proprio"] * -2}, inputs]
    with FreshVisualTokenSparsity(model, keep_ratio=0.25, selection=selection):
        references = [model.sample_action(**request) for request in requests]
    with FreshVisualTokenSparsity(model, keep_ratio=0.25, selection=selection, buffered=True) as runtime:
        for request, expected in zip(requests, references):
            if runtime.replay is not None:
                for tensor in runtime.replay.buffers:
                    if isinstance(tensor, torch.Tensor) and tensor.is_floating_point():
                        tensor.fill_(float("nan"))
            assert torch.equal(model.sample_action(**request), expected)
        with pytest.raises(ValueError):
            model.sample_action(**{**inputs, "num_steps": 0})
        assert not runtime._active and not runtime._indices
        assert torch.equal(model.sample_action(**inputs), references[0])
    runtime.close_graphs()
    assert runtime.replay is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("ratio,selection", [(1.0, "action"), (0.25, "action"), (0.25, "uniform")])
def test_cuda_graph_is_bitwise_own_eager_across_changing_observations(ratio, selection):
    with torch.device("cuda"):
        model, inputs = model_and_inputs()
    model = model.to(device="cuda", dtype=torch.bfloat16)
    inputs = {k: v.to(device="cuda", dtype=torch.bfloat16 if v.is_floating_point() else v.dtype)
              if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    requests = [inputs, {**inputs, "first_frame_latents": inputs["first_frame_latents"] + 1}, inputs]
    with FreshVisualTokenSparsity(model, keep_ratio=ratio, selection=selection):
        refs = [model.sample_action(**request) for request in requests]
    with FreshVisualTokenSparsity(model, keep_ratio=ratio, selection=selection, graph_enabled=True) as runtime:
        for request, expected in zip(requests, refs):
            assert torch.equal(model.sample_action(**request), expected)
            assert runtime.last_stats["graph_replays"] == 4
    runtime.close_graphs()


@pytest.mark.parametrize("payload", [{}, [], {"keep_ratio": 0}, {"keep_ratio": True},
    {"keep_ratio": float("nan")}, {"keep_ratio": 1.1}, {"keep_ratio": 0.1, "selection": "drift"},
    {"keep_ratio": 0.1, "refresh_every": 1}, {"keep_ratio": 0.1, "graph_dispatch": False}])
def test_invalid_options_fail_before_model_loading(monkeypatch, payload):
    from types import SimpleNamespace
    import dreamwam.policy as policy_module
    monkeypatch.setattr(policy_module, "build_model", lambda *a, **k: pytest.fail("loaded invalid options"))
    with pytest.raises(ValueError, match="fresh_visual_tokens"):
        policy_module.DreamWAMPolicy(SimpleNamespace(evaluation={}), device="cpu", fresh_visual_tokens=payload)
    assert fresh_visual_options(None) is None
