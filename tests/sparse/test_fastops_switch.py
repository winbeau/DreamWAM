"""The fused-operator switch must actually change what runs, and say so.

Wiring that is present but inert is the failure this file guards: the modules already had a
sparse path that was never consumed once, so the switch is checked by observing its effect on
the objects that consume it, not by asserting an attribute exists.
"""

from __future__ import annotations

import torch

from dreamwam.experts import ActionDiT, VideoDiT
from dreamwam.layers import DiTBlock, RMSNorm

HIDDEN = 16
FFN = 32
HEADS = 2
HEAD_DIM = 8
LAYERS = 2


def make_video_dit() -> VideoDiT:
    return VideoDiT(
        video_latent_dim=4,
        flow_latent_dim=4,
        hidden_dim=HIDDEN,
        ffn_dim=FFN,
        text_dim=8,
        freq_dim=8,
        num_heads=HEADS,
        attn_head_dim=HEAD_DIM,
        num_layers=LAYERS,
        patch_size=(1, 2, 2),
        eps=1e-6,
    )


def make_action_dit() -> ActionDiT:
    return ActionDiT(
        action_dim=4,
        hidden_dim=HIDDEN,
        ffn_dim=FFN,
        text_dim=8,
        freq_dim=8,
        num_heads=HEADS,
        attn_head_dim=HEAD_DIM,
        num_layers=LAYERS,
        eps=1e-6,
    )


def test_fast_ops_is_off_by_default():
    """The shipped numerics must be what runs unless a configuration asks otherwise."""
    assert not make_video_dit().rope_is_real
    assert not make_action_dit().rope_is_real
    assert not RMSNorm(8).fused


def test_enabling_fast_ops_switches_the_experts():
    video = make_video_dit()
    action = make_action_dit()
    video.enable_fast_ops()
    action.enable_fast_ops()
    assert video.rope_is_real and action.rope_is_real
    for expert in (video, action):
        table = expert.frame_cos if isinstance(expert, VideoDiT) else expert.cos
        assert table is not None and torch.isfinite(table).all()


def test_enabled_action_expert_hands_out_a_real_table_pair():
    """pre_dit must publish the representation apply_rope dispatches on."""
    action = make_action_dit()
    context = torch.randn(1, 3, 8)
    mask = torch.ones(1, 3, dtype=torch.bool)

    def state():
        return action.pre_dit(
            action_tokens=torch.randn(1, 5, 4),
            timestep=torch.zeros(1),
            context=context,
            context_mask=mask,
        )

    legacy = state()["freqs"]
    assert torch.is_complex(legacy) or legacy.ndim == 3
    assert not isinstance(legacy, tuple)

    action.enable_fast_ops()
    real = state()["freqs"]
    assert isinstance(real, tuple) and len(real) == 2, "a real pair must be published"
    cos, sin = real
    assert cos.shape[-1] == HEAD_DIM and sin.shape[-1] == HEAD_DIM
    assert not torch.is_complex(cos) and not torch.is_complex(sin)


def test_enabled_video_expert_publishes_real_tables():
    video = make_video_dit()
    video.enable_fast_ops()
    state = video.pre_dit(
        # Video and flow must share batch, time and space; the conditioning frame is the
        # first latent frame, so T >= 2 is required for the joint path.
        video_latents=torch.randn(1, 4, 2, 4, 4),
        flow_latents=torch.zeros(1, 4, 2, 4, 4),
        timestep=torch.zeros(1),
        context=torch.randn(1, 3, 8),
        context_mask=torch.ones(1, 3, dtype=torch.bool),
    )
    frequencies = state["freqs"]
    assert isinstance(frequencies, tuple) and len(frequencies) == 2
    expected = state["tokens"].shape[1]
    assert frequencies[0].shape[0] == expected


def test_block_rms_norm_modules_are_opt_in_per_instance():
    """A DiTBlock's norms must not be fused until something asks for it."""
    block = DiTBlock(
        hidden_dim=HIDDEN, attn_head_dim=HEAD_DIM, num_heads=HEADS, ffn_dim=FFN, eps=1e-6
    )
    norms = [m for m in block.modules() if isinstance(m, RMSNorm)]
    assert norms, "the block is expected to contain RMSNorm modules"
    assert not any(norm.fused for norm in norms)
    for norm in norms:
        norm.fused = True
    assert all(norm.fused for norm in norms)
