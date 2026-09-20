import pytest
import torch

from dreamwam.sparse.visual_cache_graphs import GraphedVisualTokenCache
from dreamwam.sparse.visual_kv_staging import ActionKVReplay, StagedVisualKVCache
from test_visual_step_cache import model_and_inputs


@torch.no_grad()
def test_staging_updates_mutated_anchor_and_dynamic_inputs_without_output_aliases():
    def compute(action, video_kv_cache, length):
        return action + video_kv_cache[0]["k"] + video_kv_cache[0]["v"]
    request = dict(action=torch.ones(2),
                   video_kv_cache=[dict(k=torch.ones(2), v=torch.ones(2) * 2)], length=2)
    replay = ActionKVReplay(compute, request, device="cpu", enabled=False)
    first = replay(request, generation=1)
    request["action"].fill_(4)
    second = replay(request, generation=1)
    assert torch.equal(first, torch.full((2,), 4.0))
    assert torch.equal(second, torch.full((2,), 7.0))
    second.fill_(100)
    # Simulate an in-place partial refresh at the same address, then a new
    # request. Correctness must follow the explicit generation, not identity.
    request["video_kv_cache"][0]["k"].fill_(8)
    assert torch.equal(replay(request, generation=2), torch.full((2,), 14.0))
    request["video_kv_cache"][0]["v"].fill_(-2)
    assert torch.equal(replay(request, generation=3), torch.full((2,), 10.0))
    assert replay.kv_stage_batches == 3 and replay.kv_skip_batches == 1
    with pytest.raises(ValueError, match="backwards"):
        replay(request, generation=2)
    with pytest.raises(ValueError, match="graph input"):
        replay({**request, "length": 3}, generation=3)


@pytest.mark.parametrize("keep", [1.0, 0.25])
@pytest.mark.parametrize("partial", [False, True])
@torch.no_grad()
def test_joint_refresh_invalidates_staging_across_partial_and_request_boundaries(keep, partial):
    model, inputs = model_and_inputs()
    inputs = {**inputs, "num_steps": 6}
    options = dict(keep_ratio=keep, refresh_every=3, graph_partial=partial, graph_enabled=False)
    reference = GraphedVisualTokenCache(model, **options)
    staged = StagedVisualKVCache(model, **options)
    original = model.mot.forward_action_with_video_cache
    for request in (inputs, {**inputs, "context": inputs["context"] * -2,
                            "first_frame_latents": inputs["first_frame_latents"] + 0.7}, inputs):
        with reference:
            expected = model.sample_action(**request)
        with staged:
            actual = model.sample_action(**request)
            assert torch.equal(actual, expected)
            assert staged.last_stats["action_kv_stage_batches"] == 2
            assert staged.last_stats["action_kv_skip_batches"] == 2
            for key, value in reference.last_stats.items():
                assert staged.last_stats[key] == value
            assert not staged.video_kv and not staged._active
        assert model.mot.forward_action_with_video_cache == original
    staged.close_graphs()
    reference.close_graphs()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph integration")
@torch.no_grad()
def test_cuda_staging_replay_preserves_changed_requests_and_actual_refresh_counts():
    # RoPE is constructed on the policy device, as in runtime.build_model.
    with torch.device("cuda"):
        model, inputs = model_and_inputs()
    inputs = {**inputs, "num_steps": 6}
    options = dict(keep_ratio=0.25, refresh_every=3, graph_partial=True)
    reference = GraphedVisualTokenCache(model, **options)
    staged = StagedVisualKVCache(model, **options)
    for request in (inputs, {**inputs, "proprio": inputs["proprio"] * 0.4}, inputs):
        with reference:
            expected = model.sample_action(**request)
        with staged:
            assert torch.equal(model.sample_action(**request), expected)
            assert staged.last_stats["action_kv_stage_batches"] == 2
            assert staged.last_stats["action_graph_replays"] == 4
            assert staged.last_stats["partial_graph_replays"] == 1
    staged.close_graphs()
    reference.close_graphs()
