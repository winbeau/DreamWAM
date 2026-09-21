"""Trace real tensors and actual routes without altering model execution."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from dreamwam.sparse.hybrid.runtime import HybridVisualRuntime
from dreamwam.sparse.hybrid.trace import HybridTrace, joint_probabilities
from test_hybrid_compact import compact
from test_visual_step_cache import model_and_inputs


def test_joint_trace_keeps_action_keys_in_softmax_and_masked_values_zero():
    query, keys = torch.zeros(1, 2, 4), torch.zeros(1, 5, 4)
    mask = torch.tensor([[True, False, True, True, True], [False, False, True, True, True]])
    probabilities = joint_probabilities(query, keys, mask, 2)
    assert torch.all(probabilities[..., 1] == 0)
    torch.testing.assert_close(probabilities.sum(-1), torch.ones(1, 2, 2))
    assert probabilities[0, 0, 0, :3].sum() == 0.5


@pytest.mark.parametrize("structure", [False, True])
def test_actual_attention_probe_and_routes_are_separate_with_bitwise_parity(structure):
    model, inputs = model_and_inputs()
    config = compact(selection="action_context")
    if structure:
        config = replace(config, reuse_mode="structure", recompute_ratio=0.5)
    with HybridVisualRuntime(model, config) as runtime:
        baseline = model.sample_action(**inputs)
        with HybridTrace(runtime, layers=(0, 1)) as trace:
            actual = model.sample_action(**inputs)
        assert torch.equal(actual, baseline)
        assert trace.grid == (3, 2, 2)
        executed = [row for row in trace.records if row["kind"] == "executed_attention"]
        probes = [row for row in trace.records if row["kind"] == "router_probe"]
        assert len(executed) == 8 and len(probes) == 3
        for record in executed:
            p = trace.arrays[record["joint_probabilities"]]
            keys = trace.arrays[record["visual_key_ids"]]
            assert p.shape == (1, 2, 4, len(keys) + 4)
            np.testing.assert_allclose(p.sum(-1), 1, atol=2e-7)
            if record["step"] == 0:
                assert keys.tolist() == list(range(12))
            else:
                assert keys.tolist() == runtime.last_stats["steps"][record["step"]]["route"]
        for record in probes:
            p = trace.arrays[record["joint_probabilities"]]
            direct = trace.arrays[record["direct"]]
            support = trace.arrays[record["support"]]
            visible = trace.arrays[record["vv_visible_keys"]]
            seeds = torch.from_numpy(trace.arrays[record["seeds"]])
            native_mask = model.mot.build_attention_mask(video_length=12, action_length=4,
                video_tokens_per_frame=4, device=seeds.device)
            np.testing.assert_array_equal(visible, native_mask[:12, :12][seeds].any(0).numpy())
            assert np.all(support[:, ~visible] == 0)
            expected = direct / direct.mean(-1, keepdims=True) + support / support.mean(-1, keepdims=True)
            np.testing.assert_allclose(trace.arrays[record["read_score"]], expected.mean(0), rtol=2e-6)
            assert p.shape[-1] == 16, "router probes full current K even when execution reads packed K"
        assert torch.equal(model.sample_action(**inputs), baseline), "instrumentation must restore all hooks"


def test_trace_rejects_graph_replay_and_invalid_coordinates():
    model, _ = model_and_inputs()
    with HybridVisualRuntime(model, compact(backend="buffered")) as runtime:
        with pytest.raises(ValueError, match="eager"):
            HybridTrace(runtime)
    with HybridVisualRuntime(model, compact()) as runtime:
        with pytest.raises(ValueError, match="layer"):
            HybridTrace(runtime, layers=(2,))
        with pytest.raises(ValueError, match="step"):
            HybridTrace(runtime, steps=(0, 0))
