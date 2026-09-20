import pytest
import torch

from dreamwam.sparse.head_stage_probe import HeadStageProbe, intervention_mask
from dreamwam.layers import scaled_dot_product_attention
from test_visual_step_cache import model_and_inputs


def test_only_target_head_future_vv_edges_change_and_zero_budget_is_nonempty():
    model, _ = model_and_inputs()
    native = model.mot.build_attention_mask(video_length=12, action_length=4,
                                            video_tokens_per_frame=4, device=torch.device("cpu"))
    mask, stats = intervention_mask(native, num_heads=2, video_length=12,
                                    tokens_per_frame=4, head=1, future_keep_ratio=0, block_size=2)
    assert torch.equal(mask[0, 0], native)
    assert torch.equal(mask[0, 1, :4], native[:4])
    assert torch.equal(mask[0, 1, 12:], native[12:])
    assert torch.equal(mask[0, 1, 4:12, :4], native[4:12, :4])
    assert not mask[0, 1, 4:12, 4:12].any()
    assert mask.any(dim=-1).all()
    assert stats["removed_pairs"] == 64
    torch.manual_seed(9)
    q, k, v = (torch.randn(1, 16, 16) for _ in range(3))
    dense = scaled_dot_product_attention(q, k, v, 2, native).view(1, 16, 2, 8)
    changed = scaled_dot_product_attention(q, k, v, 2, mask).view(1, 16, 2, 8)
    assert torch.equal(changed[:, :, 0], dense[:, :, 0])
    assert torch.equal(changed[:, 12:], dense[:, 12:])
    assert torch.equal(changed[:, :4], dense[:, :4])
    assert not torch.equal(changed[:, 4:12, 1], dense[:, 4:12, 1])


@torch.no_grad()
def test_full_budget_matches_native_and_target_phase_is_exact_and_cleans_up():
    torch.manual_seed(11)
    model, inputs = model_and_inputs()
    inputs["num_steps"] = 10
    original = model.mot._joint_self_attention
    native = model.sample_action(**inputs)
    with HeadStageProbe(model) as probe:
        full = model.sample_action(**inputs)
        assert torch.equal(full, native)
        assert probe.stats["attention_calls"] == 20
        assert probe.last_video.shape == (1, 4, 3, 4, 4)
        assert torch.equal(probe.last_video[:, :, :1], inputs["first_frame_latents"])
        for stage, expected in enumerate(([0, 1, 2], [3, 4, 5, 6], [7, 8, 9])):
            probe.configure(layer=1, head=0, stage=stage, future_keep_ratio=0)
            assert torch.isfinite(model.sample_action(**inputs)).all()
            assert probe.stats["target_steps"] == expected
            assert probe.stats["video_scheduler_steps"] == 10
        probe.configure()
        assert torch.equal(model.sample_action(**inputs), native)
        with pytest.raises(ValueError):
            model.sample_action(**{**inputs, "action_horizon": 0})
        assert probe.last_video is None and not probe.masks
    assert model.mot._joint_self_attention == original


@pytest.mark.parametrize("kwargs", [dict(layer=0), dict(layer=0, head=2, stage=0),
                                    dict(layer=2, head=0, stage=0), dict(future_keep_ratio=-0.1)])
def test_rejects_incomplete_or_out_of_range_intervention(kwargs):
    model, _ = model_and_inputs()
    with pytest.raises(ValueError):
        HeadStageProbe(model).configure(**kwargs)
