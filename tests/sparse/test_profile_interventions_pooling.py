import numpy as np
import pytest
import torch

from dreamwam.layers import scaled_dot_product_attention
from dreamwam.sparse.profile.geometry import TokenGrid
from dreamwam.sparse.profile.intervention import KeyIntervention, intervention_mask
from dreamwam.sparse.profile.pooling import PoolPlan
from dreamwam.sparse.profile.selection import combine_scores, select_tokens
from test_visual_step_cache import model_and_inputs


def test_equal_budget_baseline_indices_and_score_ablation():
    grid = TokenGrid(3, 7, 14)
    uniform = select_tokens(grid, 56, method="uniform")
    assert np.bincount(uniform // 98).tolist() == [19, 19, 18]
    assert uniform[:19].tolist() == (np.arange(19) * 98 // 19).tolist()
    score = np.arange(294, dtype=np.float64)
    top = select_tokens(grid, 56, method="score", scores=score)
    random = select_tokens(grid, 56, method="random")
    assert len(set(top)) == len(set(random)) == len(uniform) == 56
    assert np.array_equal(random, select_tokens(grid, 56, method="random"))
    future = select_tokens(grid, 20, method="score", scores=score, scope="future")
    assert future.min() >= 98 and np.bincount(future // 98).tolist() == [0, 10, 10]
    action_only = combine_scores(dict(action=score, dynamic=score[::-1]), dict(action=1, dynamic=0))
    assert np.array_equal(select_tokens(grid, 56, method="score", scores=action_only), top)


@pytest.mark.parametrize("scope", ["AV", "VV", "joint"])
def test_delete_preserves_exact_complement_and_action_keys(scope):
    native = torch.ones(8, 8, dtype=torch.bool)
    native[:6, 6:] = False
    result = intervention_mask(native, torch.tensor([1, 4]), 6, scope)
    changed = native & ~result
    rows = [6, 7] if scope == "AV" else list(range(6)) if scope == "VV" else list(range(8))
    expected = torch.zeros_like(changed)
    expected[rows, 1] = expected[rows, 4] = True
    assert torch.equal(changed, expected)
    assert torch.equal(result[:, 6:], native[:, 6:])


@pytest.mark.parametrize("operation", ["delete", "replace_value_zero"])
def test_av_only_does_not_modify_video_and_last_layer_vv_does_not_modify_action(operation):
    model, inputs = model_and_inputs()
    with KeyIntervention(model, {}, operation=operation, scope="joint") as baseline:
        expected = model.sample_action(**inputs)
        video = baseline.last_video.clone()
    target = {(3, 1): [4, 5, 6, 7]}
    with KeyIntervention(model, target, operation=operation, scope="AV") as probe:
        result = model.sample_action(**inputs)
        assert torch.equal(probe.last_video, video)
        assert not torch.equal(result, expected)
        assert len(probe.records) == 1 and probe.records[0]["count"] == 4
    with KeyIntervention(model, target, operation=operation, scope="VV") as probe:
        assert torch.equal(model.sample_action(**inputs), expected)
        assert not torch.equal(probe.last_video, video)
    assert torch.equal(model.sample_action(**inputs), expected)


def test_full_recompute_is_native_and_partial_recompute_has_no_old_request_leak():
    model, inputs = model_and_inputs()
    expected = model.sample_action(**inputs)
    with KeyIntervention(model, {(2, 0): list(range(12))}, operation="recompute", scope="joint"):
        assert torch.equal(model.sample_action(**inputs), expected)
    with KeyIntervention(model, {(2, 0): [0, 4, 8]}, operation="recompute", scope="joint") as probe:
        result = model.sample_action(**inputs)
        assert not torch.equal(result, expected)
        assert not probe.previous
        assert torch.equal(model.sample_action(**inputs), result)
        assert len(probe.records) == 1
    assert not getattr(model, "_visual_ffn_context_cache", None)


def test_pooling_preserves_mask_and_count_weight_is_exact_for_identical_keys():
    grid = TokenGrid(3, 2, 4)
    plan = PoolPlan.from_grid(grid, refined_regions=(), pool_observed=False)
    assert plan.packed_length == 12  # eight current keys and four future summaries
    native = torch.ones(5, grid.length, dtype=torch.bool)
    native[0, grid.frame_size:] = False
    key = torch.zeros(1, grid.length, 8)
    value = torch.randn_like(key)
    query = torch.randn(1, 5, 8)
    k, v, bias = plan.pack(key, value, native)
    expected = scaled_dot_product_attention(query, key, value, 2, native)
    actual = scaled_dot_product_attention(query, k, v, 2, bias)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    assert torch.isneginf(bias[0]).sum() == 4
    unweighted = PoolPlan.from_grid(grid, refined_regions=(), multiplicity="unit")
    k, v, bias = unweighted.pack(key, value, native)
    assert not torch.allclose(scaled_dot_product_attention(query, k, v, 2, bias), expected)


def test_full_refinement_is_native_order_and_bitwise_attention():
    grid = TokenGrid(3, 2, 4)
    plan = PoolPlan.from_grid(grid, refined_regions=range(len(grid.regions())))
    key, value, query = (torch.randn(1, grid.length, 8) for _ in range(3))
    mask = torch.ones(grid.length, grid.length, dtype=torch.bool)
    k, v, bias = plan.pack(key, value, mask)
    assert torch.equal(k, key) and torch.equal(v, value)
    assert torch.equal(scaled_dot_product_attention(query, k, v, 2, bias),
                       scaled_dot_product_attention(query, key, value, 2, mask))


def test_pooling_rejects_mixed_visibility_and_unlabelled_mixed_cache_age():
    grid = TokenGrid(3, 2, 4)
    plan = PoolPlan.from_grid(grid, refined_regions=())
    key = value = torch.ones(1, grid.length, 8)
    mask = torch.ones(5, grid.length, dtype=torch.bool)
    mask[0, 8] = False
    with pytest.raises(ValueError, match="visibility"):
        plan.pack(key, value, mask)
    mask[:] = True
    ages = torch.zeros(grid.length, dtype=torch.long)
    ages[8] = 1
    with pytest.raises(ValueError, match="fresh/stale"):
        plan.pack(key, value, mask, ages=ages)


def test_manual_pool_groups_cannot_bypass_camera_boundary():
    grid = TokenGrid(1, 1, 4)
    plan = PoolPlan(((0, 2), (1, 3)), grid, "count")
    with pytest.raises(ValueError, match="camera boundary"):
        plan.pack(torch.ones(1, 4, 8), torch.ones(1, 4, 8), torch.ones(1, 4, dtype=torch.bool))


def test_intervention_cannot_target_action_keys_or_wrap_negative_indices():
    for target in [-1, 6]:
        with pytest.raises(ValueError, match="visual positions"):
            intervention_mask(torch.ones(8, 8, dtype=torch.bool), torch.tensor([target]), 6, "AV")
