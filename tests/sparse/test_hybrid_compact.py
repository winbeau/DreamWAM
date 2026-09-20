"""Compact keys must change physical work, preserve visibility, and have valid ages."""

from dataclasses import replace

import pytest
import torch

import dreamwam.sparse.hybrid.execution as execution
from dreamwam.sparse.hybrid.runtime import HybridVisualRuntime
from test_hybrid_runtime import config
from test_visual_step_cache import model_and_inputs


def compact(operations=("dense", "sparse", "reuse", "sparse"), **kwargs):
    kwargs.setdefault("frame_quota", "balanced")
    return config(operations, read_mode="compact", read_ratio=0.5, diagnostics="trace", **kwargs)


@pytest.mark.parametrize("method", ["uniform", "drift", "action_drift"])
def test_compact_shapes_swaps_and_cache_age(monkeypatch, method):
    model, inputs = model_and_inputs()
    runtime = HybridVisualRuntime(model, compact(selection=method))
    original = execution.scaled_dot_product_attention
    dimensions = []
    def observe(query, key, value, heads, mask):
        dimensions.append((query.shape[1], key.shape[1]))
        assert key.shape == value.shape
        assert mask.any(dim=-1).all()
        assert mask[-4:, -4:].all(), "all action keys remain jointly normalized"
        return original(query, key, value, heads, mask)
    monkeypatch.setattr(execution, "scaled_dot_product_attention", observe)
    with runtime:
        result = model.sample_action(**inputs)
        assert torch.isfinite(result).all()
        trace = runtime.last_stats["steps"]
        for previous, row in zip(trace, trace[1:]):
            new_keys = set(row["route"]) - set(previous["route"])
            assert new_keys <= set(row["query"] or [])
            assert len(row["route"]) == len(set(row["route"])) == 6
            assert [sum(i // 4 == frame for i in row["route"]) for frame in range(3)] == [2, 2, 2]
        assert torch.equal(model.sample_action(**inputs), result)
    assert dimensions[:6] == [(7, 10), (7, 10), (4, 10), (4, 10), (7, 10), (7, 10)]
    assert runtime.last_stats["computed_video_token_layers"] == 36
    assert not runtime.state.packed and not runtime.state.kv


def test_reuse_reads_existing_packed_buffers_without_repacking(monkeypatch):
    model, inputs = model_and_inputs()
    with HybridVisualRuntime(model, compact(("dense", "reuse", "reuse", "reuse"),
                                            selection="uniform")) as runtime:
        original = runtime._call
        pointers = []
        def call(op, request):
            if op == "reuse":
                pointers.append([layer["k"].data_ptr() for layer in request["kv"]])
                assert request["kv"] is runtime.state.packed
                assert all(layer["k"].shape[1] == 6 for layer in request["kv"])
            return original(op, request)
        monkeypatch.setattr(runtime, "_call", call)
        model.sample_action(**inputs)
    assert pointers[0] == pointers[1] == pointers[2]


@pytest.mark.parametrize("all_video", [True, False])
def test_compact_math_matches_independent_full_shape_masked_reference(monkeypatch, all_video):
    model, inputs = model_and_inputs()
    model.mot.action_attends_all_video = all_video
    conf = compact(selection="uniform")
    with HybridVisualRuntime(model, conf):
        actual = model.sample_action(**inputs)
    native = execution.scaled_dot_product_attention
    full_mask = model.mot.build_attention_mask(video_length=12, action_length=4,
                                                video_tokens_per_frame=4, device="cpu")
    # Uniform balanced reads are [0,2 | 4,6 | 8,10], and updates [0 | 4 | 8].
    # Expand both axes into the ORIGINAL grid and use original visibility,
    # independently of the compact mask passed by the implementation.
    key_index = torch.tensor([0, 2, 4, 6, 8, 10, 12, 13, 14, 15])
    def reference(query, key, value, heads, mask):
        q_index = (torch.tensor([0, 4, 8, 12, 13, 14, 15]) if query.shape[1] == 7
                   else torch.arange(12, 16))
        q = query.new_zeros((query.shape[0], 16, query.shape[-1])).index_copy(1, q_index, query)
        k = key.new_zeros((key.shape[0], 16, key.shape[-1])).index_copy(1, key_index, key)
        v = value.new_zeros((value.shape[0], 16, value.shape[-1])).index_copy(1, key_index, value)
        supported = torch.zeros(16, dtype=torch.bool)
        supported[key_index] = True
        reference_mask = full_mask & supported[None, :]
        assert torch.equal(mask, reference_mask.index_select(0, q_index).index_select(1, key_index))
        return native(q, k, v, heads, reference_mask).index_select(1, q_index)
    monkeypatch.setattr(execution, "scaled_dot_product_attention", reference)
    with HybridVisualRuntime(model, conf):
        expected = model.sample_action(**inputs)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


def test_sparse_commit_preserves_unselected_rows_and_dense_reanchors(monkeypatch):
    model, inputs = model_and_inputs()
    with HybridVisualRuntime(model, compact(("dense", "sparse", "dense", "reuse"))) as runtime:
        commit = runtime.state.commit_sparse
        checked = []
        def observe(result, current, query, route, step, **kwargs):
            untouched = torch.ones(current.shape[1], dtype=torch.bool)
            untouched[query] = False
            before = [[layer[name][:, untouched].clone() for name in ("k", "v")] for layer in runtime.state.kv]
            hidden = runtime.state.output[:, untouched].clone()
            commit(result, current, query, route, step, **kwargs)
            for old, layer in zip(before, runtime.state.kv):
                for tensor, name in zip(old, ("k", "v")):
                    assert torch.equal(tensor, layer[name][:, untouched])
            assert torch.equal(hidden, runtime.state.output[:, untouched])
            checked.append(True)
        monkeypatch.setattr(runtime.state, "commit_sparse", observe)
        model.sample_action(**inputs)
        assert runtime.last_stats["steps"][2]["feature_age_max"] == 0
    assert checked == [True]


def test_full_budget_compact_control_is_native():
    model, inputs = model_and_inputs()
    expected = model.sample_action(**inputs)
    conf = replace(compact(("dense", "sparse", "sparse", "sparse")), recompute_ratio=1, read_ratio=1)
    with HybridVisualRuntime(model, conf):
        assert torch.equal(model.sample_action(**inputs), expected)


@pytest.mark.parametrize("method", ["uniform", "drift", "action_drift"])
def test_balanced_queries_match_across_read_modes_given_identical_causal_inputs(monkeypatch, method):
    from dreamwam.sparse.hybrid.selection import select
    model, inputs = model_and_inputs()
    conf = compact(("dense", "sparse", "reuse", "reuse"), selection=method)
    checked = []
    with HybridVisualRuntime(model, conf) as runtime:
        original = runtime._sparse_request
        def compare(video, action, selection):
            full = replace(conf, read_mode="full", read_ratio=1)
            expected = select(full, model, video, action, runtime.state)
            assert torch.equal(selection.query, expected.query)
            assert len(expected.route) == 12 and len(selection.route) == 6
            checked.append(True)
            return original(video, action, selection)
        monkeypatch.setattr(runtime, "_sparse_request", compare)
        model.sample_action(**inputs)
    assert checked == [True]
