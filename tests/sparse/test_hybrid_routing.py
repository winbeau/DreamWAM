"""Decision support is backward, causal, explicit-cost and independently ablatable."""

from dataclasses import replace

import pytest
import torch

from dreamwam.sparse.hybrid import HybridConfig
from dreamwam.sparse.hybrid.routing import support_mass
from dreamwam.sparse.hybrid.runtime import HybridVisualRuntime
from dreamwam.sparse.hybrid.selection import scores, select
from test_hybrid_compact import compact
from test_visual_step_cache import model_and_inputs


def test_support_propagates_seed_weight_to_visible_keys_not_consumers():
    query = torch.zeros(1, 2, 2)
    key = torch.zeros(1, 3, 2)
    weights = torch.tensor([[0.8, 0.2]])
    mask = torch.tensor([[True, True, False], [False, True, True]])
    actual = support_mass(query, key, weights, num_heads=1, mask=mask)
    torch.testing.assert_close(actual, torch.tensor([[0.4, 0.5, 0.1]]))
    key[:, 2] = 100  # A masked key must not leak into seed zero's support.
    torch.testing.assert_close(support_mass(query, key, weights, num_heads=1, mask=mask), actual)


@pytest.mark.parametrize("method", ["action", "action_context", "visual_context"])
def test_current_probes_costs_cache_contract_and_request_isolation(method):
    model, inputs = model_and_inputs()
    conf = compact(selection=method)
    with HybridVisualRuntime(model, conf) as runtime:
        expected = model.sample_action(**inputs)
        stats = runtime.last_stats
        assert stats["video_key_probe_rows"] == 3 * 12
        expected_queries = 0 if method == "action" else 3 * (12 if method == "visual_context" else 2)
        assert stats["video_query_probe_rows"] == expected_queries
        assert stats["action_probe_rows"] == (0 if method == "visual_context" else 3 * 4)
        assert stats["computed_video_token_layers"] == 36  # Probes are counted separately.
        for before, after in zip(stats["steps"], stats["steps"][1:]):
            assert set(after["route"]) - set(before["route"]) <= set(after["query"] or [])
            assert len(after["route"]) == len(set(after["route"])) == 6
        model.sample_action(**{**inputs, "context": inputs["context"] + 3})
        assert torch.equal(model.sample_action(**inputs), expected)


def test_zero_context_weight_is_exact_action_only_ablation():
    model, inputs = model_and_inputs()
    with HybridVisualRuntime(model, compact(selection="action")) as action:
        expected = model.sample_action(**inputs)
    with HybridVisualRuntime(model, compact(selection="action_context", context_weight=0)) as context:
        assert torch.equal(model.sample_action(**inputs), expected)
    assert action.last_stats["video_query_probe_rows"] == context.last_stats["video_query_probe_rows"] == 0
    assert [s["route"] for s in action.last_stats["steps"]] == [s["route"] for s in context.last_stats["steps"]]


def test_read_ranking_survives_zero_drift_and_does_not_read_old_keys(monkeypatch):
    model, inputs = model_and_inputs()
    conf = compact(selection="action_context")
    with HybridVisualRuntime(model, conf) as runtime:
        original = runtime._sparse_request
        def inspect(video, action, selected):
            old_reference = runtime.state.reference
            runtime.state.reference = video["tokens"].clone()
            ranking = scores(conf, model, video, action, runtime.state)
            assert torch.count_nonzero(ranking.query) == 0
            assert ranking.read.std() > 0
            # Current projection, not a hidden dependency on stale cached K.
            old_keys = [layer["k"].clone() for layer in runtime.state.kv]
            for layer in runtime.state.kv:
                layer["k"].fill_(float("nan"))
            again = scores(conf, model, video, action, runtime.state)
            assert torch.equal(again.read, ranking.read)
            for layer, key in zip(runtime.state.kv, old_keys):
                layer["k"].copy_(key)
            runtime.state.reference = old_reference
            full = select(replace(conf, read_mode="full", read_ratio=1), model, video, action, runtime.state)
            assert torch.equal(full.query, selected.query)
            return original(video, action, selected)
        monkeypatch.setattr(runtime, "_sparse_request", inspect)
        model.sample_action(**inputs)


def test_new_options_are_strict_and_legacy_identity_is_unchanged():
    conf = compact(selection="action_context")
    assert HybridConfig.from_mapping(conf.describe()).policy_hash == conf.policy_hash
    for change in (dict(context_weight=-1), dict(context_weight=True),
                   dict(support_seed_ratio=0), dict(support_seed_ratio=float("nan"))):
        with pytest.raises(ValueError):
            replace(conf, **change)
    assert "context_weight" not in compact(selection="action_drift").describe()["selection"]
