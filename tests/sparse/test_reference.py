"""Correctness tests for the Sparse-WAM operator reference.

These run without a checkpoint: they exercise mask semantics, the AV normalisation, route
construction and the masked/gather backends against explicit references on small tensors.
The full-model parity control (dense vs split vs full budget) lives in
``scripts/sparse/check_parity.py`` because it needs the released weights.

    .venv/bin/python -m pytest tests/sparse -q
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from dreamwam.sparse import (
    SparseConfig,
    build_route,
    build_token_layout,
    sparse_joint_attention,
    video_legality,
)
from dreamwam.sparse.av import action_attention_with_mass
from dreamwam.sparse.layout import block_hit_mask

TOKENS_PER_FRAME = 4
NUM_FRAMES = 3
NUM_HEADS = 2
HEAD_DIM = 4
WIDTH = NUM_HEADS * HEAD_DIM
ACTION_TOKENS = 3
BLOCK_SIZE = 2


def make_layout(device="cpu"):
    return build_token_layout(
        video_length=TOKENS_PER_FRAME * NUM_FRAMES,
        tokens_per_frame=TOKENS_PER_FRAME,
        block_size=BLOCK_SIZE,
        device=torch.device(device),
    )


def tensors(batch=1, seed=0, device="cpu"):
    generator = torch.Generator(device=device).manual_seed(seed)
    video_length = TOKENS_PER_FRAME * NUM_FRAMES

    def rand(length):
        return torch.randn(
            batch, length, WIDTH, generator=generator, device=device, dtype=torch.float32
        )

    return rand(video_length), rand(video_length), rand(video_length), rand(
        ACTION_TOKENS
    ), rand(ACTION_TOKENS), rand(ACTION_TOKENS)


# --- mask semantics ---------------------------------------------------------------


def test_legality_matches_native_mask():
    """Frame 0 sees only frame 0; every other frame sees the whole video."""
    layout = make_layout()
    mask = video_legality(layout)
    frame = torch.arange(layout.video_length) // TOKENS_PER_FRAME
    for query in range(layout.video_length):
        for key in range(layout.video_length):
            expected = (frame[query].item() != 0) or (frame[key].item() == 0)
            assert bool(mask[query, key]) is expected


def test_frame_zero_rows_always_have_a_legal_key():
    layout = make_layout()
    mask = video_legality(layout)
    assert bool(mask[:TOKENS_PER_FRAME, :TOKENS_PER_FRAME].all())
    assert not bool(mask[:TOKENS_PER_FRAME, TOKENS_PER_FRAME:].any())


def test_block_hit_mask_marks_exactly_the_selected_tokens():
    layout = make_layout()
    ids = torch.tensor([[[0, 2]]])  # [B=1, H=1, m=2]
    membership = block_hit_mask(layout, ids)
    expected = torch.zeros(layout.video_length, dtype=torch.bool)
    for block in (0, 2):
        expected[layout.future_block_keys[block]] = True
    assert torch.equal(membership[0, 0], expected)
    assert not bool(membership[0, 0, :TOKENS_PER_FRAME].any())


def test_block_hit_mask_ignores_padding():
    layout = make_layout()
    ids = torch.tensor([[[1, -1]]])
    membership = block_hit_mask(layout, ids)
    selected = membership[0, 0].nonzero().flatten().tolist()
    assert selected == layout.future_block_keys[1].tolist()


# --- AV mass ----------------------------------------------------------------------


def test_action_attention_matches_joint_softmax():
    """The manual action branch must reproduce the fused joint computation."""
    query_action, key_video, value_video, key_action, value_action = (
        tensors()[3],
        tensors()[1],
        tensors()[2],
        tensors()[4],
        tensors()[5],
    )
    output, mass = action_attention_with_mass(
        query_action=query_action,
        key_video=key_video,
        value_video=value_video,
        key_action=key_action,
        value_action=value_action,
        num_heads=NUM_HEADS,
    )
    scale = HEAD_DIM**-0.5
    q = query_action.view(1, ACTION_TOKENS, NUM_HEADS, HEAD_DIM).transpose(1, 2)
    kv = key_video.view(1, -1, NUM_HEADS, HEAD_DIM).transpose(1, 2)
    ka = key_action.view(1, ACTION_TOKENS, NUM_HEADS, HEAD_DIM).transpose(1, 2)
    vv = value_video.view(1, -1, NUM_HEADS, HEAD_DIM).transpose(1, 2)
    va = value_action.view(1, ACTION_TOKENS, NUM_HEADS, HEAD_DIM).transpose(1, 2)
    logits = torch.cat([q @ kv.transpose(-1, -2), q @ ka.transpose(-1, -2)], dim=-1) * scale
    probs = torch.softmax(logits.float(), dim=-1)
    reference = (
        probs[..., : key_video.shape[1]] @ vv.float()
        + probs[..., key_video.shape[1] :] @ va.float()
    ).to(output.dtype)
    assert torch.allclose(output, reference.transpose(1, 2).reshape_as(output), atol=1e-5)
    # Mass over the video keys only; the action keys carry the rest.  The mass is
    # aggregated over action queries, so the per-head total is bounded by the number of
    # action queries rather than by 1.
    assert torch.allclose(mass, probs[..., : key_video.shape[1]].sum(-2), atol=1e-5)
    assert float(mass.amax(dim=-1).max()) <= 1.0 + 1e-5
    assert float(mass.sum(-1).max()) <= ACTION_TOKENS + 1e-4


def test_action_mass_uses_action_keys_in_the_denominator():
    """A degenerate action key must reduce the video mass, proving no renormalisation."""
    query_action, key_video, value_video, key_action, value_action = (
        tensors()[3],
        tensors()[1],
        tensors()[2],
        tensors()[4],
        tensors()[5],
    )
    _, mass = action_attention_with_mass(
        query_action=query_action,
        key_video=key_video,
        value_video=value_video,
        key_action=key_action,
        value_action=value_action,
        num_heads=NUM_HEADS,
    )
    peaked = key_action.clone()
    peaked[:, 0, :] += 50.0
    _, mass_peaked = action_attention_with_mass(
        query_action=query_action,
        key_video=key_video,
        value_video=value_video,
        key_action=peaked,
        value_action=value_action,
        num_heads=NUM_HEADS,
    )
    assert float(mass_peaked.mean()) < float(mass.mean())
    assert not torch.allclose(mass, mass_peaked)


# --- config -----------------------------------------------------------------------


def test_unknown_option_is_rejected():
    with pytest.raises(ValueError, match="unknown sparse option"):
        SparseConfig.from_mapping({"enable": True})


def test_stage_mapping_covers_every_step():
    config = SparseConfig.from_mapping({"enabled": True, "num_stages": 3})
    seen = {config.stage_of(step, 10) for step in range(10)}
    assert seen == {0, 1, 2}


def test_blocks_for_clamps_to_available_blocks():
    layout = make_layout()
    config = SparseConfig.from_mapping(
        {"enabled": True, "future_ratio": 1.0, "block_size": BLOCK_SIZE}
    )
    blocks = config.blocks_for(
        num_heads=NUM_HEADS, stage=0, num_future_blocks=layout.num_future_blocks
    )
    assert blocks == (layout.num_future_blocks,) * NUM_HEADS


def test_per_head_blocks_are_validated():
    with pytest.raises(ValueError, match="budget has"):
        SparseConfig.from_mapping({"enabled": True, "head_blocks": [1, 2, 3]}).blocks_for(
            num_heads=NUM_HEADS, stage=0, num_future_blocks=4
        )


# --- routing ----------------------------------------------------------------------


@pytest.mark.parametrize("selection", ["all", "av", "av_context", "recency", "uniform"])
def test_route_always_keeps_the_conditioning_frame(selection):
    layout = make_layout()
    config = SparseConfig.from_mapping(
        {
            "enabled": True,
            "selection": selection,
            "block_size": BLOCK_SIZE,
            "future_ratio": 0.5,
        }
    )
    query_video, key_video, _, _, _, _ = tensors()
    av_mass = torch.rand(1, NUM_HEADS, layout.video_length)
    route = build_route(
        layout=layout,
        config=config,
        num_heads=NUM_HEADS,
        step_index=0,
        num_steps=10,
        av_mass=av_mass,
        query_video=query_video,
        key_video=key_video,
    )
    assert bool(route.membership[:, :, :TOKENS_PER_FRAME].all())
    assert bool(route.valid[:, :, :TOKENS_PER_FRAME].all())


def test_route_respects_the_budget_per_head():
    layout = make_layout()
    config = SparseConfig.from_mapping(
        {
            "enabled": True,
            "selection": "av",
            "block_size": BLOCK_SIZE,
            "head_blocks": [0, 2],
        }
    )
    query_video, key_video, _, _, _, _ = tensors()
    route = build_route(
        layout=layout,
        config=config,
        num_heads=NUM_HEADS,
        step_index=0,
        num_steps=10,
        av_mass=torch.rand(1, NUM_HEADS, layout.video_length),
        query_video=query_video,
        key_video=key_video,
    )
    assert int(route.blocks[0, 0]) == 0
    assert int(route.blocks[0, 1]) == 2
    head0_keys = route.valid[0, 0].sum().item()
    assert head0_keys == TOKENS_PER_FRAME


def test_av_selection_prefers_the_high_mass_block():
    layout = make_layout()
    config = SparseConfig.from_mapping(
        {
            "enabled": True,
            "selection": "av",
            "block_size": BLOCK_SIZE,
            "head_blocks": [1, 1],
        }
    )
    query_video, key_video, _, _, _, _ = tensors()
    av_mass = torch.zeros(1, NUM_HEADS, layout.video_length)
    wanted = layout.future_block_keys[2]
    av_mass[:, :, wanted] = 1.0
    route = build_route(
        layout=layout,
        config=config,
        num_heads=NUM_HEADS,
        step_index=0,
        num_steps=10,
        av_mass=av_mass,
        query_video=query_video,
        key_video=key_video,
    )
    for head in range(NUM_HEADS):
        assert int(route.blocks[0, head]) == 1
        chosen = set(route.keys[0, head, TOKENS_PER_FRAME:].tolist())
        assert chosen == set(wanted.tolist())


def test_low_anchor_confidence_triggers_the_declared_fallback():
    layout = make_layout()
    config = SparseConfig.from_mapping(
        {
            "enabled": True,
            "selection": "av",
            "block_size": BLOCK_SIZE,
            "head_blocks": [1, 1],
            "min_anchor_mass": 0.99,
            "fallback": "recency",
        }
    )
    query_video, key_video, _, _, _, _ = tensors()
    av_mass = torch.full((1, NUM_HEADS, layout.video_length), 1.0 / layout.video_length)
    route = build_route(
        layout=layout,
        config=config,
        num_heads=NUM_HEADS,
        step_index=0,
        num_steps=10,
        av_mass=av_mass,
        query_video=query_video,
        key_video=key_video,
    )
    assert bool(route.fallback.all())
    last_block = layout.future_block_keys[-1]
    for head in range(NUM_HEADS):
        assert set(route.keys[0, head, TOKENS_PER_FRAME:].tolist()) == set(
            last_block.tolist()
        )


def test_full_budget_route_keeps_every_legal_pair():
    layout = make_layout()
    config = SparseConfig.from_mapping(
        {"enabled": True, "selection": "all", "block_size": BLOCK_SIZE}
    )
    route = build_route(
        layout=layout,
        config=config,
        num_heads=NUM_HEADS,
        step_index=0,
        num_steps=10,
    )
    future = layout.video_length - TOKENS_PER_FRAME
    assert bool(route.membership[:, :, TOKENS_PER_FRAME:].all())
    assert route.density == pytest.approx(1.0)
    assert future > 0


# --- execution backends -----------------------------------------------------------


def reference_video_attention(query, key, value, membership, layout):
    """Dense attention over exactly the routed keys, high precision."""
    mask = membership.unsqueeze(2) & video_legality(layout).view(
        1, 1, layout.video_length, layout.video_length
    )
    return F.scaled_dot_product_attention(
        query.view(1, -1, NUM_HEADS, HEAD_DIM).transpose(1, 2),
        key.view(1, -1, NUM_HEADS, HEAD_DIM).transpose(1, 2),
        value.view(1, -1, NUM_HEADS, HEAD_DIM).transpose(1, 2),
        attn_mask=mask,
    ).transpose(1, 2).reshape_as(query)


@pytest.mark.parametrize("backend", ["masked", "gather"])
@pytest.mark.parametrize("selection", ["all", "av", "recency", "uniform"])
def test_backends_agree_with_the_reference(backend, selection):
    layout = make_layout()
    config = SparseConfig.from_mapping(
        {
            "enabled": True,
            "selection": selection,
            "backend": backend,
            "block_size": BLOCK_SIZE,
            "future_ratio": 0.5,
        }
    )
    query_video, key_video, value_video, query_action, key_action, value_action = tensors()
    # Use the same anchor mass the implementation derives, so the independently built
    # reference route and the executed route are the same object of comparison.
    _, av_mass = action_attention_with_mass(
        query_action=query_action,
        key_video=key_video,
        value_video=value_video,
        key_action=key_action,
        value_action=value_action,
        num_heads=NUM_HEADS,
    )
    route = build_route(
        layout=layout,
        config=config,
        num_heads=NUM_HEADS,
        step_index=0,
        num_steps=10,
        av_mass=av_mass,
        query_video=query_video,
        key_video=key_video,
    )
    video_out, action_out, stats, _ = sparse_joint_attention(
        query_video=query_video,
        key_video=key_video,
        value_video=value_video,
        query_action=query_action,
        key_action=key_action,
        value_action=value_action,
        num_heads=NUM_HEADS,
        layout=layout,
        config=config,
        step_index=0,
        num_steps=10,
        action_mask=None,
    )
    expected = reference_video_attention(
        query_video, key_video, value_video, route.membership, layout
    )
    assert torch.allclose(video_out, expected, atol=1e-5), (
        f"{backend}/{selection} diverged from the routed reference"
    )
    assert stats["density"] <= 1.0


def test_masked_and_gather_backends_agree_on_the_same_config():
    """The two execution backends must be interchangeable for identical routing."""
    layout = make_layout()
    outputs = {}
    for backend in ("masked", "gather"):
        config = SparseConfig.from_mapping(
            {
                "enabled": True,
                "selection": "av",
                "backend": backend,
                "block_size": BLOCK_SIZE,
                "future_ratio": 0.5,
            }
        )
        query_video, key_video, value_video, query_action, key_action, value_action = (
            tensors()
        )
        video_out, action_out, stats, _ = sparse_joint_attention(
            query_video=query_video,
            key_video=key_video,
            value_video=value_video,
            query_action=query_action,
            key_action=key_action,
            value_action=value_action,
            num_heads=NUM_HEADS,
            layout=layout,
            config=config,
            step_index=0,
            num_steps=10,
        )
        outputs[backend] = (video_out, action_out, stats)
    assert torch.allclose(outputs["masked"][0], outputs["gather"][0], atol=1e-5)
    assert torch.allclose(outputs["masked"][1], outputs["gather"][1], atol=1e-6)
    assert outputs["masked"][2]["density"] == pytest.approx(
        outputs["gather"][2]["density"]
    )


def test_gather_backend_skips_future_keys_when_budget_is_zero():
    layout = make_layout()
    config = SparseConfig.from_mapping(
        {
            "enabled": True,
            "selection": "all",
            "backend": "gather",
            "block_size": BLOCK_SIZE,
            "future_ratio": 0.0,
        }
    )
    query_video, key_video, value_video, query_action, key_action, value_action = tensors()
    video_out, _, stats, _ = sparse_joint_attention(
        query_video=query_video,
        key_video=key_video,
        value_video=value_video,
        query_action=query_action,
        key_action=key_action,
        value_action=value_action,
        num_heads=NUM_HEADS,
        layout=layout,
        config=config,
        step_index=0,
        num_steps=10,
    )
    assert stats["kmax"] == TOKENS_PER_FRAME
    # Frame-0 rows are unaffected by dropping future keys because they cannot see them.
    dense = reference_video_attention(
        query_video,
        key_video,
        value_video,
        torch.zeros(1, NUM_HEADS, layout.video_length, dtype=torch.bool).index_fill_(
            2, layout.first_frame_keys, True
        ),
        layout,
    )
    assert torch.allclose(video_out[:, :TOKENS_PER_FRAME], dense[:, :TOKENS_PER_FRAME], atol=1e-5)


def test_action_rows_ignore_the_video_route():
    """A->V is dense regardless of how the VV route is restricted."""
    layout = make_layout()
    outputs = []
    for ratio in (1.0, 0.0):
        config = SparseConfig.from_mapping(
            {
                "enabled": True,
                "selection": "all",
                "backend": "masked",
                "block_size": BLOCK_SIZE,
                "future_ratio": ratio,
            }
        )
        query_video, key_video, value_video, query_action, key_action, value_action = (
            tensors()
        )
        _, action_out, _, _ = sparse_joint_attention(
            query_video=query_video,
            key_video=key_video,
            value_video=value_video,
            query_action=query_action,
            key_action=key_action,
            value_action=value_action,
            num_heads=NUM_HEADS,
            layout=layout,
            config=config,
            step_index=0,
            num_steps=10,
        )
        outputs.append(action_out)
    assert torch.allclose(outputs[0], outputs[1], atol=1e-6)


def test_batch_elements_get_independent_routes():
    layout = make_layout()
    batch = 2
    config = SparseConfig.from_mapping(
        {
            "enabled": True,
            "selection": "av",
            "block_size": BLOCK_SIZE,
            "head_blocks": [1, 1],
        }
    )
    query_video, key_video, _, query_action, key_action, value_action = tensors(batch=batch)
    value_video = tensors(batch=batch)[2]
    av_mass = torch.zeros(batch, NUM_HEADS, layout.video_length)
    av_mass[0, :, layout.future_block_keys[0]] = 1.0
    av_mass[1, :, layout.future_block_keys[2]] = 1.0
    route = build_route(
        layout=layout,
        config=config,
        num_heads=NUM_HEADS,
        step_index=0,
        num_steps=10,
        av_mass=av_mass,
        query_video=query_video,
        key_video=key_video,
    )
    assert set(route.keys[0, 0, TOKENS_PER_FRAME:].tolist()) != set(
        route.keys[1, 0, TOKENS_PER_FRAME:].tolist()
    )
    video_out, _, _, _ = sparse_joint_attention(
        query_video=query_video,
        key_video=key_video,
        value_video=value_video,
        query_action=query_action,
        key_action=key_action,
        value_action=value_action,
        num_heads=NUM_HEADS,
        layout=layout,
        config=config,
        step_index=0,
        num_steps=10,
    )
    assert video_out.shape == query_video.shape
    assert torch.isfinite(video_out).all()


def test_disabled_config_raises_rather_than_running_dense():
    layout = make_layout()
    with pytest.raises(ValueError, match="disabled"):
        build_route(
            layout=layout,
            config=SparseConfig(),
            num_heads=NUM_HEADS,
            step_index=0,
            num_steps=10,
        )


# --- route amortization (M3) ------------------------------------------------------


def test_reused_route_is_the_one_executed():
    """Passing a route must execute it verbatim, and skip anchor extraction."""
    layout = make_layout()
    config = SparseConfig.from_mapping(
        {
            "enabled": True,
            "selection": "av",
            "backend": "masked",
            "block_size": BLOCK_SIZE,
            "head_blocks": [1, 2],
        }
    )
    query_video, key_video, value_video, query_action, key_action, value_action = tensors()
    _, av_mass = action_attention_with_mass(
        query_action=query_action,
        key_video=key_video,
        value_video=value_video,
        key_action=key_action,
        value_action=value_action,
        num_heads=NUM_HEADS,
    )
    first = build_route(
        layout=layout,
        config=config,
        num_heads=NUM_HEADS,
        step_index=0,
        num_steps=10,
        av_mass=av_mass,
        query_video=query_video,
        key_video=key_video,
    )
    video_out, action_out, stats, returned = sparse_joint_attention(
        query_video=query_video,
        key_video=key_video,
        value_video=value_video,
        query_action=query_action,
        key_action=key_action,
        value_action=value_action,
        num_heads=NUM_HEADS,
        layout=layout,
        config=config,
        step_index=0,
        num_steps=10,
        route=first,
    )
    assert returned is first
    assert stats["anchors_recomputed"] is False, "a reused route must not rebuild anchors"
    expected = reference_video_attention(
        query_video, key_video, value_video, first.membership, layout
    )
    assert torch.allclose(video_out, expected, atol=1e-5)


def test_reused_route_matches_the_freshly_built_one():
    """Amortizing the route across layers must not change what that layer computes."""
    layout = make_layout()
    config = SparseConfig.from_mapping(
        {
            "enabled": True,
            "selection": "av",
            "backend": "masked",
            "block_size": BLOCK_SIZE,
            "head_blocks": [1, 2],
        }
    )
    query_video, key_video, value_video, query_action, key_action, value_action = tensors()
    _, av_mass = action_attention_with_mass(
        query_action=query_action,
        key_video=key_video,
        value_video=value_video,
        key_action=key_action,
        value_action=value_action,
        num_heads=NUM_HEADS,
    )
    route = build_route(
        layout=layout,
        config=config,
        num_heads=NUM_HEADS,
        step_index=0,
        num_steps=10,
        av_mass=av_mass,
        query_video=query_video,
        key_video=key_video,
    )
    common = dict(
        query_video=query_video,
        key_video=key_video,
        value_video=value_video,
        query_action=query_action,
        key_action=key_action,
        value_action=value_action,
        num_heads=NUM_HEADS,
        layout=layout,
        config=config,
        step_index=0,
        num_steps=10,
    )
    fresh_video, fresh_action, _, built = sparse_joint_attention(**common)
    reused_video, reused_action, _, _ = sparse_joint_attention(**common, route=route)
    # The rebuilt route must equal the reused one, and the outputs must agree with it.
    assert torch.equal(built.keys, route.keys)
    assert torch.equal(built.membership, route.membership)
    assert torch.allclose(fresh_video, reused_video, atol=1e-5)
    # The action branch differs only by kernel (manual vs fused), so allow bf16-level slack.
    assert torch.allclose(fresh_action, reused_action, atol=1e-4)


def test_anchor_refresh_is_validated_and_defaults_to_per_layer():
    assert SparseConfig().anchor_refresh == "layer"
    with pytest.raises(ValueError, match="anchor_refresh"):
        SparseConfig.from_mapping({"enabled": True, "anchor_refresh": "sometimes"})
    with pytest.raises(ValueError, match="anchor_layer"):
        SparseConfig.from_mapping({"enabled": True, "anchor_layer": -1})
    for scope in ("layer", "step", "request"):
        assert (
            SparseConfig.from_mapping(
                {"enabled": True, "anchor_refresh": scope}
            ).anchor_refresh
            == scope
        )


# --- route lifetime (M3 amortization) ---------------------------------------------


def test_route_scope_key_matches_the_declared_refresh():
    from dreamwam.sparse.runtime import route_scope_key

    per_layer = SparseConfig.from_mapping({"enabled": True, "anchor_refresh": "layer"})
    per_step = SparseConfig.from_mapping({"enabled": True, "anchor_refresh": "step"})
    per_request = SparseConfig.from_mapping({"enabled": True, "anchor_refresh": "request"})
    assert route_scope_key(per_layer, 3) is None
    assert route_scope_key(per_step, 3) == (3,)
    assert route_scope_key(per_step, 4) == (4,)
    # A request-scoped route must be the SAME key for every step, otherwise it silently
    # degrades to step scope - the bug this test exists to prevent.
    assert route_scope_key(per_request, 0) == route_scope_key(per_request, 7)


def test_dense_prefix_covers_layers_before_the_anchor_layer():
    from dreamwam.sparse.runtime import is_dense_prefix

    late = SparseConfig.from_mapping(
        {"enabled": True, "anchor_refresh": "step", "anchor_layer": 2}
    )
    assert is_dense_prefix(late, 0) and is_dense_prefix(late, 1)
    assert not is_dense_prefix(late, 2)
    # Per-layer routing has no prefix: every layer builds its own route.
    per_layer = SparseConfig.from_mapping({"enabled": True, "anchor_refresh": "layer"})
    assert not is_dense_prefix(per_layer, 0)


def test_route_is_built_once_per_scope_and_reused():
    from dreamwam.sparse.runtime import should_build_route

    config = SparseConfig.from_mapping({"enabled": True, "anchor_refresh": "step"})
    assert should_build_route(config, 0, None) is True
    assert should_build_route(config, 1, object()) is False
    assert should_build_route(config, 29, object()) is False
    per_layer = SparseConfig.from_mapping({"enabled": True, "anchor_refresh": "layer"})
    assert should_build_route(per_layer, 0, object()) is True
    late = SparseConfig.from_mapping(
        {"enabled": True, "anchor_refresh": "step", "anchor_layer": 2}
    )
    assert should_build_route(late, 0, None) is False  # dense prefix, not a build
    assert should_build_route(late, 2, None) is True


# --- conditioning-frame compression (video branch budget) -------------------------


def test_full_conditional_ratio_reproduces_the_original_route():
    """The default must not change anything: ratio 1.0 keeps the whole conditioning frame."""
    layout = make_layout()
    base = {
        "enabled": True,
        "selection": "recency",
        "block_size": BLOCK_SIZE,
        "head_blocks": [1, 1],
    }
    default = build_route(
        layout=layout, config=SparseConfig.from_mapping(base), num_heads=NUM_HEADS,
        step_index=0, num_steps=10,
    )
    explicit = build_route(
        layout=layout,
        config=SparseConfig.from_mapping({**base, "conditional_keep_ratio": 1.0}),
        num_heads=NUM_HEADS,
        step_index=0,
        num_steps=10,
    )
    assert torch.equal(default.keys, explicit.keys)
    assert default.density == pytest.approx(explicit.density)


def test_conditional_ratio_shrinks_the_frame0_budget():
    layout = make_layout()
    route = build_route(
        layout=layout,
        config=SparseConfig.from_mapping(
            {
                "enabled": True,
                "selection": "recency",
                "block_size": BLOCK_SIZE,
                "future_ratio": 0.0,
                "conditional_keep_ratio": 0.5,
            }
        ),
        num_heads=NUM_HEADS,
        step_index=0,
        num_steps=10,
    )
    # TOKENS_PER_FRAME = 4, so half is 2 conditioning keys for future rows.
    assert int(route.valid[0, 0].sum()) == 2
    assert int(route.keys.shape[-1]) == 2


def test_conditional_ratio_never_drops_every_conditioning_key():
    """A zero ratio must still leave one key; an empty row would produce NaN attention."""
    layout = make_layout()
    route = build_route(
        layout=layout,
        config=SparseConfig.from_mapping(
            {
                "enabled": True,
                "selection": "recency",
                "block_size": BLOCK_SIZE,
                "future_ratio": 0.0,
                "conditional_keep_ratio": 0.0,
            }
        ),
        num_heads=NUM_HEADS,
        step_index=0,
        num_steps=10,
    )
    assert int(route.valid[0, 0].sum()) >= 1
    assert bool(route.valid[0].any(dim=-1).all())


def test_compressed_conditioning_frame_is_ranked_by_action_relevance():
    """With anchors available, the kept conditioning keys are the ones the action looks at."""
    layout = make_layout()
    query_video, key_video, _, _, _, _ = tensors()
    av_mass = torch.zeros(1, NUM_HEADS, layout.video_length)
    wanted = [1, 3]
    av_mass[:, :, wanted] = 5.0
    route = build_route(
        layout=layout,
        config=SparseConfig.from_mapping(
            {
                "enabled": True,
                "selection": "av",
                "block_size": BLOCK_SIZE,
                "future_ratio": 0.0,
                "conditional_keep_ratio": 0.5,
            }
        ),
        num_heads=NUM_HEADS,
        step_index=0,
        num_steps=10,
        av_mass=av_mass,
        query_video=query_video,
        key_video=key_video,
    )
    for head in range(NUM_HEADS):
        assert set(route.keys[0, head].tolist()) == set(wanted)


def test_frame0_rows_stay_dense_over_their_own_frame_when_it_is_compressed():
    """Both backends must agree that frame-0 queries still see the whole conditioning frame."""
    layout = make_layout()
    outputs = {}
    for backend in ("masked", "gather"):
        config = SparseConfig.from_mapping(
            {
                "enabled": True,
                "selection": "recency",
                "backend": backend,
                "block_size": BLOCK_SIZE,
                "future_ratio": 0.0,
                "conditional_keep_ratio": 0.25,
            }
        )
        query_video, key_video, value_video, query_action, key_action, value_action = (
            tensors()
        )
        video_out, _, _, _ = sparse_joint_attention(
            query_video=query_video,
            key_video=key_video,
            value_video=value_video,
            query_action=query_action,
            key_action=key_action,
            value_action=value_action,
            num_heads=NUM_HEADS,
            layout=layout,
            config=config,
            step_index=0,
            num_steps=10,
        )
        outputs[backend] = video_out
        # Frame-0 rows equal a fully dense attention over frame 0 alone.
        dense = F.scaled_dot_product_attention(
            query_video[:, :TOKENS_PER_FRAME].view(1, -1, NUM_HEADS, HEAD_DIM).transpose(1, 2),
            key_video[:, :TOKENS_PER_FRAME].view(1, -1, NUM_HEADS, HEAD_DIM).transpose(1, 2),
            value_video[:, :TOKENS_PER_FRAME].view(1, -1, NUM_HEADS, HEAD_DIM).transpose(1, 2),
        ).transpose(1, 2).reshape(1, TOKENS_PER_FRAME, WIDTH)
        assert torch.allclose(
            video_out[:, :TOKENS_PER_FRAME], dense, atol=1e-5
        ), f"{backend}: frame-0 rows must be independent of the conditioning budget"
    assert torch.allclose(outputs["masked"], outputs["gather"], atol=1e-5)


def test_conditional_ratio_is_validated():
    with pytest.raises(ValueError, match="conditional_keep_ratio"):
        SparseConfig.from_mapping({"enabled": True, "conditional_keep_ratio": 1.5})


def test_padding_never_erases_a_legitimate_key():
    """Regression: padding is clamped to index 0, so writing its False directly into the
    membership table used to erase key 0 whenever a head had a smaller budget than the widest
    head. That is the per-head-budget case M1 exists to produce, and the masked backend reads
    membership, so the failure was silent and only visible against an independent reference."""
    layout = make_layout()
    route = build_route(
        layout=layout,
        config=SparseConfig.from_mapping(
            {
                "enabled": True,
                "selection": "recency",
                "block_size": BLOCK_SIZE,
                "head_blocks": [0, 2],  # head 0 keeps no future block -> padding exists
            }
        ),
        num_heads=NUM_HEADS,
        step_index=0,
        num_steps=10,
    )
    assert bool(route.valid[0, 0].any()), "head 0 must keep its conditioning keys"
    assert bool(route.membership[0, 0].any()), "membership must not be emptied by padding"
    # Every valid gathered key must be a member.
    for head in range(NUM_HEADS):
        for key, ok in zip(route.keys[0, head].tolist(), route.valid[0, head].tolist()):
            if ok:
                assert bool(route.membership[0, head, key]), (
                    f"head {head}: key {key} is executed but absent from membership"
                )


def test_block_hit_mask_agrees_with_the_route_it_describes():
    """The standalone helper and the route builder must produce the same membership.

    Two implementations of the same table is how the padding defect survived: the helper and
    the router were both wrong in the same way, so comparing either against the other proved
    nothing. They are compared here against the executed route's own keys instead.
    """
    layout = make_layout()
    ids = torch.tensor([[[2, -1]]])  # one real block followed by padding
    helper = block_hit_mask(layout, ids)
    selected = layout.future_block_keys[2]
    for token in range(layout.video_length):
        expected = bool((selected == token).any())
        assert bool(helper[0, 0, token]) is expected, f"token {token} membership disagrees"
    assert not bool(helper[0, 0, :TOKENS_PER_FRAME].any()), (
        "the helper covers future blocks only; frame 0 is the caller's addition"
    )


def test_routing_does_not_synchronise_the_device():
    """A device->host read in the route builder makes the whole denoising loop uncapturable
    by CUDA graphs, which is the overhead this project is removing. Asserted statically
    because a synchronisation is invisible in the output, only in the timing."""
    import ast
    from pathlib import Path

    import dreamwam.sparse.attention as attention
    import dreamwam.sparse.reuse as reuse
    import dreamwam.sparse.routing as routing

    offenders = []
    for module in (routing, attention, reuse):
        tree = ast.parse(Path(module.__file__).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in {"item", "cpu", "numpy", "tolist"}:
                    offenders.append((module.__name__, node.lineno, node.func.attr))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in {"bool", "float", "int"}:
                    offenders.append((module.__name__, node.lineno, node.func.id))
    assert not offenders, (
        "the sparse hot path must not read device tensors into Python; found "
        f"{offenders}. Each of these synchronises once per layer-step, so 300 per request."
    )


def test_fallback_blending_is_unconditional_and_correct():
    """The unconditional where must still prefer the fallback ordering when it fires."""
    layout = make_layout()
    query_video, key_video, _, _, _, _ = tensors()
    flat = torch.full((1, NUM_HEADS, layout.video_length), 1.0 / layout.video_length)
    route = build_route(
        layout=layout,
        config=SparseConfig.from_mapping(
            {
                "enabled": True,
                "selection": "av",
                "block_size": BLOCK_SIZE,
                "head_blocks": [1, 1],
                "min_anchor_mass": 0.99,
                "fallback": "recency",
            }
        ),
        num_heads=NUM_HEADS,
        step_index=0,
        num_steps=10,
        av_mass=flat,
        query_video=query_video,
        key_video=key_video,
    )
    assert bool(route.fallback.all())
    last = set(layout.future_block_keys[-1].tolist())
    for head in range(NUM_HEADS):
        assert set(route.keys[0, head, TOKENS_PER_FRAME:].tolist()) == last
