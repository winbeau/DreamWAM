"""Numerical and lifecycle checks for the actual selective-FFN wrapper."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from dreamwam.sparse.ffn_context_cache import VisualFFNContextCache


class TinyJoint(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(setting="joint")
        self.video_expert = nn.Module()
        block = nn.Module()
        block.ffn = nn.Sequential(nn.Linear(3, 5), nn.GELU(), nn.Linear(5, 3))
        self.video_expert.blocks = nn.ModuleList([block])
        self.action_ffn = nn.Linear(3, 3)
        self.eval()

    @torch.no_grad()
    def sample_action(self, inputs, fail=False):
        outputs = [self.video_expert.blocks[0].ffn(value) for value in inputs]
        if fail:
            raise RuntimeError("simulated request failure")
        return outputs


def test_full_budget_matches_dense_and_restores_parameter_identity():
    model = TinyJoint()
    inputs = [torch.randn(2, 7, 3) for _ in range(3)]
    dense = model.sample_action(inputs)
    original = model.video_expert.blocks[0].ffn.forward
    parameters = {key: value.clone() for key, value in model.state_dict().items()}
    with VisualFFNContextCache(model, keep_ratio=1.0) as cache:
        actual = model.sample_action(inputs)
        assert all(torch.equal(x, y) for x, y in zip(dense, actual))
        assert cache.last_stats["computed_rows"] == 42
        assert cache.last_stats["selective_calls"] == 0
    assert model.video_expert.blocks[0].ffn.forward == original
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in parameters.items())


def test_recomputed_rows_match_dense_and_other_rows_keep_their_reference():
    model = TinyJoint()
    x = torch.ones(2, 5, 3)
    y = x.clone()
    y[0, 3] = 4
    y[1, 1] = 5
    dense = model.sample_action([x, y])
    with VisualFFNContextCache(model, keep_ratio=0.2) as cache:
        actual = model.sample_action([x, y])
        torch.testing.assert_close(actual[1], dense[1])
        assert cache.last_stats == dict(dense_calls=1, selective_calls=1, computed_rows=12, total_rows=20)
        assert not cache.references


def test_input_reference_advances_only_for_rows_recomputed():
    model = TinyJoint()
    x = torch.ones(1, 4, 3)
    y = x.clone()
    y[:, 0] = 10
    y[:, 1] = 2
    z = y.clone()
    with VisualFFNContextCache(model, keep_ratio=0.25):
        actual = model.sample_action([x, y, z])
    dense = model.sample_action([x, y, z])
    # Step 1 updates row 0. Row 1 has accumulated drift and must win step 2,
    # even though the current input is identical to the previous step's input.
    torch.testing.assert_close(actual[1][:, 1], dense[0][:, 1])
    torch.testing.assert_close(actual[2][:, 1], dense[2][:, 1])
    torch.testing.assert_close(actual[1][:, 0], dense[1][:, 0])


def test_new_requests_and_failed_requests_never_reuse_old_references():
    model = TinyJoint()
    a, b = torch.randn(1, 6, 3), torch.randn(1, 6, 3)
    expected = model.sample_action([b])[0]
    original_action = model.action_ffn.forward
    with VisualFFNContextCache(model, keep_ratio=0.0) as cache:
        actual = model.sample_action([a, b])
        assert torch.equal(actual[0], actual[1])
        with pytest.raises(RuntimeError, match="simulated"):
            model.sample_action([a], fail=True)
        assert not cache.references and not cache._active
        assert torch.equal(model.sample_action([b])[0], expected)
        assert model.action_ffn.forward == original_action


@pytest.mark.parametrize("ratio", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_ratio_rejected(ratio):
    with pytest.raises(ValueError):
        VisualFFNContextCache(TinyJoint(), keep_ratio=ratio)
