"""Independent mask oracle, real SDPA extents and stage/replay isolation."""

import pytest
import torch

from dreamwam.layers import scaled_dot_product_attention
from dreamwam.sparse.head_stage_execution import (
    HeadStageExecution, HeadStageGraph, compact_attention, make_plan, stage_index,
)
from dreamwam.sparse.visual_cache_graphs import TensorTreeReplay
from test_head_stage_allocations import module as allocation
from test_visual_step_cache import model_and_inputs


def native_mask(prefix, video, actions, device="cpu"):
    native = torch.ones(video + actions, video + actions, dtype=torch.bool, device=device)
    native[:prefix, prefix:] = False
    native[prefix:video, video:] = False
    return native


@pytest.mark.parametrize("budgets", [[0, 0, 0, 0], [2, 2, 2, 2], [1, 1, 1, 1], [0, 2, 1, 2]])
@torch.no_grad()
def test_compact_matches_independent_joint_mask_and_submits_shorter_keys(budgets, monkeypatch):
    torch.manual_seed(12)
    native = native_mask(2, 6, 3)
    plan = make_plan(native, budgets, video_length=6, tokens_per_frame=2, block_size=2)
    oracle = allocation.allocated_mask(native, budgets, video_length=6, tokens_per_frame=2, block_size=2)
    assert torch.equal(plan.mask, oracle)
    video = tuple(torch.randn(2, 6, 32, dtype=torch.float64) for _ in range(3))
    action = tuple(torch.randn(2, 3, 32, dtype=torch.float64) for _ in range(3))
    expected = scaled_dot_product_attention(*(torch.cat((v, a), 1) for v, a in zip(video, action)), 4, oracle)
    calls = []
    original = torch.nn.functional.scaled_dot_product_attention

    def observed(q, k, v, **kwargs):
        calls.append((q.shape[1], q.shape[2], k.shape[2]))
        return original(q, k, v, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", observed)
    actual = compact_attention(video, action, heads=4, plan=plan)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    assert calls[0] == (4, 5, 9), "action/conditioning rows retain all native joint keys"
    assert sorted(calls[1:]) == sorted((budgets.count(k), 4, 2 + k * 2) for k in set(budgets))
    assert sum(h * q * k for h, q, k in calls) == plan.matrix_pairs
    assert plan.matrix_pairs < 4 * 9 * 9


def test_plan_rejects_preexisting_mask_changes_and_invalid_budgets():
    native = native_mask(2, 6, 3)
    for budgets in ([True], [0.5], [-1], [3], []):
        with pytest.raises(ValueError):
            make_plan(native, budgets, video_length=6, tokens_per_frame=2, block_size=2)
    native[6, 5] = False
    with pytest.raises(ValueError, match="native Joint"):
        make_plan(native, [1], video_length=6, tokens_per_frame=2, block_size=2)
    assert [stage_index(i, 10) for i in range(10)] == [0, 0, 0, 1, 1, 1, 1, 2, 2, 2]
    for step, total in [(-1, 10), (10, 10), (0, 9)]:
        with pytest.raises(ValueError):
            stage_index(step, total)


@torch.no_grad()
def test_stage_graph_buffering_retains_budget_changes_and_request_isolation():
    torch.manual_seed(5)
    model, inputs = model_and_inputs()
    inputs.update(num_steps=10, first_frame_latents=torch.randn(1, 4, 1, 4, 14))
    profile = [[[0, 2], [2, 0], [0, 2]] for _ in range(model.mot.num_layers)]
    controller = HeadStageExecution(model.mot, profile, backend="compact")
    graph = HeadStageGraph(model, enabled=False)
    original_sample, original_forward, original_attention = model.sample_action, model.mot.forward, model.mot._joint_self_attention
    native = model.sample_action(**inputs)
    for request in (inputs, {**inputs, "proprio": inputs["proprio"] + 0.2}, inputs):
        with controller:
            expected = model.sample_action(**request)
            with graph:
                actual = model.sample_action(**request)
                assert torch.equal(actual, expected)
                assert graph.last_stats == dict(stage_calls=[3, 4, 3], graph_replays=0)
        assert model.sample_action == original_sample
        assert model.mot.forward == original_forward
        assert model.mot._joint_self_attention == original_attention
    assert torch.equal(model.sample_action(**inputs), native)
    assert set(graph.graphs) == {0, 1, 2}
    graph.close_graphs()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an admitted CUDA window")
@pytest.mark.parametrize("budgets", [[14] * 24, [7] * 24, [0, 14] * 12])
@torch.no_grad()
def test_released_bf16_geometry_and_cuda_replay(budgets):
    torch.manual_seed(29)
    native = native_mask(98, 294, 32, "cuda")
    plan = make_plan(native, budgets, video_length=294, tokens_per_frame=98)
    video = tuple(torch.randn(1, 294, 3072, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    action = tuple(torch.randn(1, 32, 3072, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    oracle = allocation.allocated_mask(native, budgets, video_length=294, tokens_per_frame=98)
    expected = scaled_dot_product_attention(*(torch.cat((v, a), 1) for v, a in zip(video, action)), 24, oracle)
    actual = compact_attention(video, action, heads=24, plan=plan)
    delta = (actual.float() - expected.float())
    assert delta.norm() / expected.float().norm() <= 0.006
    assert delta.abs().max() <= 0.016
    replay = TensorTreeReplay(lambda video, action: compact_attention(video, action, heads=24, plan=plan),
                              dict(video=video, action=action), device="cuda", enabled=True)
    for offset in (0.0, 0.1, 0.0):
        changed = tuple(v + offset for v in video)
        reference = compact_attention(changed, action, heads=24, plan=plan)
        assert torch.equal(replay(dict(video=changed, action=action)), reference)
    assert replay.graph is not None and replay.replays == 3
