"""Pin neuron selection, compact weights, residual reuse, and group lifetime."""

import pytest
import torch
from torch.nn import functional as F

from dreamwam.sparse.ffn_neuron_cache import VisualFFNNeuronCache
from test_ffn_context_cache import TinyJoint


def test_compact_neuron_update_preserves_unselected_anchor_contribution():
    model = TinyJoint()
    x, y = torch.randn(1, 6, 3), torch.randn(1, 6, 3)
    up, activation, down = model.video_expert.blocks[0].ffn
    with torch.no_grad():
        h0 = activation(up(x))
        anchor = down(h0)
        scores = h0.float().abs().sum((0, 1)) * down.weight.float().norm(dim=0)
        index = scores.argsort(descending=True, stable=True)[:2]
        h1 = activation(F.linear(y, up.weight[index], up.bias[index]))
        expected = anchor + F.linear(h1 - h0[..., index], down.weight[:, index])
    cache = VisualFFNNeuronCache(model, keep_ratio=0.4, group_size=3)
    cache.prepare()
    with cache:
        actual = model.sample_action([x, y, y])
        torch.testing.assert_close(actual[1], expected)
        assert torch.equal(actual[1], actual[2]), "one group must reuse one mask and anchor"
        assert cache.last_stats["mask_builds"] == 1
        assert cache.last_stats["dense_calls"] == 1
        assert cache.last_stats["selective_calls"] == 2
        assert cache.last_stats["gathered_weight_elements"] == 2 * 3 + 3 * 2
        assert not cache.references


def test_group_boundary_and_request_boundary_refresh_dense_anchor():
    model = TinyJoint()
    inputs = [torch.randn(1, 6, 3) for _ in range(5)]
    dense = model.sample_action(inputs)
    cache = VisualFFNNeuronCache(model, keep_ratio=0.4, group_size=2)
    cache.prepare()
    with cache:
        actual = model.sample_action(inputs)
        for step in (0, 2, 4):
            assert torch.equal(actual[step], dense[step])
        assert cache.last_stats["mask_builds"] == 3
        assert cache.last_stats["dense_calls"] == 3
        assert torch.equal(model.sample_action([inputs[1]])[0], dense[1])
        assert cache.last_stats["mask_builds"] == 1


def test_full_neuron_budget_is_bitwise_dense():
    model = TinyJoint()
    inputs = [torch.randn(2, 7, 3) for _ in range(3)]
    dense = model.sample_action(inputs)
    with VisualFFNNeuronCache(model, keep_ratio=1.0) as cache:
        actual = model.sample_action(inputs)
        assert all(torch.equal(a, b) for a, b in zip(actual, dense))
        assert cache.last_stats["mask_builds"] == 0


@pytest.mark.parametrize("group_size", [0, -1, True, 1.5])
def test_invalid_group_rejected(group_size):
    with pytest.raises(ValueError):
        VisualFFNNeuronCache(TinyJoint(), keep_ratio=0.1, group_size=group_size)
