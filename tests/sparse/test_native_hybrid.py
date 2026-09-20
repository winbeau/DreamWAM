from dataclasses import replace

import pytest
import torch

from dreamwam.sparse.hybrid.config import HybridConfig
from dreamwam.sparse.hybrid.native_config import NativeRoutingConfig
from dreamwam.sparse.hybrid.native_packing import compile_pool_geometry, pool_read_count, pooled_layer
from dreamwam.sparse.hybrid.native_scores import native_layer_scores, native_read_route
from dreamwam.sparse.hybrid.runtime import HybridVisualRuntime
from dreamwam.sparse.hybrid.schedule import Schedule
from dreamwam.sparse.profile.geometry import TokenGrid
from dreamwam.sparse.profile.pooling import PoolPlan
from dreamwam.sparse.profile.signals import joint_probabilities, token_signals
from test_hybrid_compact import compact
from test_visual_step_cache import model_and_inputs


def native_config(operations=("dense", "sparse", "reuse", "sparse"), **options):
    count = options.pop("read_count", 6)
    ratio = options.pop("recompute_ratio", 0.25)
    return HybridConfig(Schedule(len(operations), operations), selection="native",
        read_mode="compact", read_count=count, frame_quota="balanced", recompute_ratio=ratio,
        diagnostics="trace", native_routing=NativeRoutingConfig(**options))


def test_configuration_roundtrip_exact_budgets_and_unsupported_combinations():
    config = native_config()
    assert config.budgets(12, 4) == (3, 6)
    assert HybridConfig.from_mapping(config.describe()).policy_hash == config.policy_hash
    assert "native_routing" not in compact().describe()
    for options in (dict(layerwise=True), dict(packing="pool", observed="full")):
        with pytest.raises(ValueError, match="Dense/reuse"):
            native_config(**options)
    with pytest.raises(ValueError, match="count or read.keep_ratio"):
        HybridConfig.from_mapping({**config.describe(), "read": dict(mode="compact", keep_count=6, keep_ratio=0.5)})
    with pytest.raises(ValueError, match="actual grid"):
        replace(config, read_count=13).budgets(12, 4)
    with pytest.raises(ValueError, match="calibrated weights"):
        NativeRoutingConfig(signal="fusion")


@pytest.mark.parametrize("signal,offline", [("action", "action"), ("value_norm", "value_norm"),
    ("value_action", "value_action"), ("dynamic", "value_video_time"),
    ("visual_context", "visual_context"), ("action_context", "action_context_support")])
def test_native_scores_match_independent_complete_joint_profile(signal, offline):
    grid = TokenGrid(3, 2, 2)
    model, _ = model_and_inputs()
    mask = model.mot.build_attention_mask(video_length=12, action_length=4, video_tokens_per_frame=4, device="cpu")
    video = tuple(torch.randn(1, 12, 8) for _ in range(3))
    action = tuple(torch.randn(1, 4, 8) for _ in range(3))
    q, k, v = [torch.cat((video[i], action[i]), dim=1)[0].reshape(16, 2, 4).transpose(0, 1) for i in range(3)]
    expected = token_signals(joint_probabilities(q, k, mask), v, grid)[offline].mean(dim=0)
    config = NativeRoutingConfig(signal=signal, heads=2)
    actual = native_layer_scores(config, video, action, num_heads=2, frame_size=4, mask=mask)
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
    fusion = replace(config, signal="fusion", weights=((signal, 1.), ("visual_context" if signal != "visual_context" else "action", 0.)))
    normalized = native_layer_scores(fusion, video, action, num_heads=2, frame_size=4, mask=mask)
    torch.testing.assert_close(normalized, actual / actual.mean().clamp_min(1e-30))


@pytest.mark.parametrize("signal", ["action", "value_action", "dynamic", "action_context"])
def test_full_budget_is_native_and_read_ranking_does_not_change_recompute_budget(signal):
    model, inputs = model_and_inputs()
    expected = model.sample_action(**inputs)
    full = native_config(("dense", "sparse", "sparse", "sparse"), signal=signal, read_count=12, recompute_ratio=1)
    with HybridVisualRuntime(model, full):
        assert torch.equal(model.sample_action(**inputs), expected)
    config = native_config(signal=signal, recompute="uniform")
    with HybridVisualRuntime(model, config) as runtime:
        original = model.sample_action(**inputs)
        rows = runtime.last_stats["steps"]
        assert [row["native_score_age"] for row in rows] == [0, 1, 2, 3]
        for before, after in zip(rows, rows[1:]):
            assert set(after["route"]) - set(before["route"]) <= set(after["query"] or [])
            assert len(after["route"]) == len(set(after["route"])) == 6
            if after["effective_op"] == "sparse":
                assert after["query"] == [0, 4, 8]
                assert set(after["query"]) <= set(after["route"])
        model.sample_action(**{**inputs, "context": inputs["context"] + 2})
        assert torch.equal(model.sample_action(**inputs), original)
    assert runtime.state.native_scores is None
    assert runtime.last_stats["selector_fallback_events"] == 0


@pytest.mark.parametrize("multiplicity", ["count", "unit"])
def test_real_grid_pool_matches_independent_partition_and_preserves_visibility(multiplicity):
    grid = TokenGrid(3, 7, 14)
    nv = grid.length
    mask = torch.ones(nv + 4, nv + 4, dtype=torch.bool)
    mask[nv:, grid.frame_size:nv] = False
    config = NativeRoutingConfig(signal="dynamic", observed="full", packing="pool", multiplicity=multiplicity)
    geometry = compile_pool_geometry(grid, mask, 0.25, "cpu")
    assert geometry["packed_length"] == pool_read_count(grid, 0.25) == 198
    key, value = torch.randn(1, nv, 8), torch.randn(1, nv, 8)
    packed, bias, route, members, sizes = pooled_layer(key, value, torch.arange(nv).float(), torch.tensor(True), mask, geometry, config)
    groups = tuple(tuple(row[row >= 0].tolist()) for row in members)
    assert sorted(i for group in groups for i in group) == list(range(nv))
    assert route.tolist() == [group[0] for group in groups]
    assert sizes.tolist() == [len(group) for group in groups]
    assert all(len(groups[i]) == 1 and groups[i][0] == i for i in range(98))
    reference = PoolPlan(groups, grid, multiplicity)
    k, v, visual_bias = reference.pack(key, value, mask[nv:, :nv], ages=torch.zeros(nv, dtype=torch.long))
    torch.testing.assert_close(packed["k"], k)
    torch.testing.assert_close(packed["v"], v)
    torch.testing.assert_close(bias[:, :198], visual_bias)
    assert (bias[:, 198:] == 0).all(), "all action keys retain original visibility and no multiplicity bias"
    bad = mask.clone()
    group = next(group for group in grid.regions() if len(group) > 1 and group[0] >= 98)
    bad[nv, group[0]] = True
    with pytest.raises(ValueError, match="different native AV visibility"):
        compile_pool_geometry(grid, bad, 0.25, "cpu")


@pytest.mark.parametrize("packing", ["hard", "pool"])
def test_full_refinement_and_layerwise_full_reads_degenerate_to_full_key_reuse(packing):
    model, inputs = model_and_inputs()
    operations = ("dense", "reuse", "reuse", "reuse")
    with HybridVisualRuntime(model, replace(compact(operations, selection="uniform"), read_ratio=1)):
        expected = model.sample_action(**inputs)
    config = native_config(operations, packing=packing, observed="full", refine_fraction=1,
                           read_count=12, layerwise=True)
    with HybridVisualRuntime(model, config) as runtime:
        assert torch.equal(model.sample_action(**inputs), expected)
        assert all(len(layer) == 12 for layer in runtime.last_stats["steps"][0]["route"])
        assert runtime.last_stats["computed_video_token_layers"] == 24


def test_nonfinite_score_falls_back_to_uniform_and_is_audited(monkeypatch):
    model, inputs = model_and_inputs()
    ops = ("dense", "reuse", "reuse", "reuse")
    with HybridVisualRuntime(model, compact(ops, selection="uniform")):
        expected = model.sample_action(**inputs)
    with HybridVisualRuntime(model, native_config(ops)) as runtime:
        original = runtime.executor.dense
        def invalid(**kwargs):
            result = original(**kwargs)
            result["native_valid"].fill_(False)
            return result
        monkeypatch.setattr(runtime.executor, "dense", invalid)
        assert torch.equal(model.sample_action(**inputs), expected)
        assert runtime.last_stats["selector_fallback_events"] == 1
        assert runtime.last_stats["steps"][0]["selector_fallback"] is True
    config = NativeRoutingConfig()
    route = native_read_route(config, torch.full((12,), torch.inf), torch.tensor(False), count=6, frame_size=4, frames=3)
    assert route.tolist() == [0, 2, 4, 6, 8, 10]


@pytest.mark.parametrize("placement", ["shared", "layerwise", "pool", "structure"])
@pytest.mark.parametrize("device,backend", [("cpu", "buffered"), ("cuda", "cuda_graph")])
def test_native_routing_graphs_replay_changed_inputs_and_overwrite_every_buffer(device, backend, placement):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires fresh separate GPU admission")
    model, inputs = model_and_inputs()
    model = model.to(device)
    inputs = {name: value.to(device) if isinstance(value, torch.Tensor) else value for name, value in inputs.items()}
    if placement in ("layerwise", "pool"):
        config = native_config(("dense", "reuse", "dense", "reuse"), layerwise=placement == "layerwise",
            packing="pool" if placement == "pool" else "hard", observed="full" if placement == "pool" else "score",
            refine_fraction=0.5, read_count=10 if placement == "pool" else 6, signal="dynamic")
    else:
        config = native_config(signal="value_action", recompute_ratio=0.5 if placement == "structure" else 0.25)
        if placement == "structure":
            config = replace(config, reuse_mode="structure")
    runtime = HybridVisualRuntime(model, replace(config, backend=backend))
    for request in (inputs, {**inputs, "context": inputs["context"] - 3,
                            "first_frame_latents": inputs["first_frame_latents"] * -2}, inputs):
        with HybridVisualRuntime(model, config):
            expected = model.sample_action(**request)
        for _, replay in runtime.dispatch.entries.values():
            for buffer in replay.buffers:
                if isinstance(buffer, torch.Tensor):
                    buffer.fill_(float("nan") if buffer.is_floating_point() else False if buffer.dtype == torch.bool else -17)
        with runtime:
            actual = model.sample_action(**request)
            assert torch.equal(actual, expected)
            assert runtime.last_stats["graph_replays"] == (4 if device == "cuda" else 0)
        assert runtime.state.native_scores is None and not runtime.state.kv
    assert len(runtime.graph_stats()) == (2 if placement == "structure" else 3)
    assert all(row["captured"] == (device == "cuda") for row in runtime.graph_stats().values())
    runtime.close_graphs()
