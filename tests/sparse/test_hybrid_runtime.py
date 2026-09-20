"""Full Joint sampling: compatibility, physical work, and cache isolation."""

from dataclasses import replace

import pytest
import torch

from dreamwam.sparse.action_guided_visual_token_cache import ActionGuidedVisualTokenCache
from dreamwam.sparse.hybrid import HybridConfig
from dreamwam.sparse.hybrid.runtime import HybridVisualRuntime
from dreamwam.sparse.hybrid.schedule import Schedule
from test_visual_step_cache import model_and_inputs


def config(operations=("dense", "reuse", "sparse", "reuse"), **kwargs):
    return HybridConfig(Schedule(len(operations), operations), recompute_ratio=0.25, **kwargs)


@pytest.mark.parametrize("method,weight", [("drift", 0), ("action_drift", 0), ("action_drift", 1)])
def test_full_read_matches_legacy_sampling(method, weight):
    model, inputs = model_and_inputs()
    with ActionGuidedVisualTokenCache(model, keep_ratio=0.25, refresh_every=2, guidance_weight=weight):
        expected = model.sample_action(**inputs)
    runtime = HybridVisualRuntime(model, config(selection=method, guidance_weight=weight))
    with runtime:
        actual = model.sample_action(**inputs)
        assert torch.equal(actual, expected)
        assert runtime.last_stats["computed_video_token_layers"] == 30
        assert not runtime.state.kv and runtime.state.output is None
        assert torch.equal(model.sample_action(**inputs), expected)


def test_all_dense_and_full_budget_sparse_are_native():
    model, inputs = model_and_inputs()
    expected = model.sample_action(**inputs)
    for conf in (config(("dense",) * 4),
                 replace(config(("dense", "sparse", "sparse", "sparse")), recompute_ratio=1)):
        with HybridVisualRuntime(model, conf) as runtime:
            assert torch.equal(model.sample_action(**inputs), expected)
            assert runtime.last_stats["dense_steps"] == 4


def test_explicit_last_step_refresh_physical_work_and_scheduler_counts(monkeypatch):
    model, inputs = model_and_inputs()
    lengths, scheduler_calls = [], []
    handle = model.video_expert.blocks[0].self_attn.q.register_forward_hook(
        lambda module, args, output: lengths.append(args[0].shape[1]))
    for name in ("video", "action"):
        scheduler = getattr(model, name + "_scheduler")
        original = scheduler.step
        def step(*args, _original=original, _name=name, **kwargs):
            scheduler_calls.append(_name)
            return _original(*args, **kwargs)
        monkeypatch.setattr(scheduler, "step", step)
    with HybridVisualRuntime(model, config(("dense", "reuse", "reuse", "sparse"),
                                          diagnostics="trace")) as runtime:
        model.sample_action(**inputs)
    handle.remove()
    assert lengths == [12, 3]
    assert scheduler_calls == ["video", "action"] * 4
    assert runtime.last_stats["action_layer_updates"] == 8
    rows = runtime.last_stats["steps"]
    assert [r["q_rows"] for r in rows] == [12, 0, 0, 3]
    assert rows[2]["route_age"] == 2
    assert rows[3]["feature_age_max"] == 3
    assert all(r["video_t"] is not None and r["route_hash"] for r in rows)
    phases = runtime.last_stats["phase_timings"]
    assert sum(row["phase"] == "execute_reuse" for row in phases) == 2
    assert all(row["cpu_seconds"] >= 0 for row in phases)
    assert all(row["cuda_stream_span_seconds"] is None for row in phases)


def test_failed_partial_call_does_not_commit_and_next_request_is_clean(monkeypatch):
    model, inputs = model_and_inputs()
    runtime = HybridVisualRuntime(model, config())
    with runtime:
        expected = model.sample_action(**inputs)
        original = runtime.executor.sparse
        def fail(**kwargs):
            before = [layer["k"].clone() for layer in runtime.state.kv]
            original(**kwargs)
            assert all(torch.equal(x, y["k"]) for x, y in zip(before, runtime.state.kv))
            raise RuntimeError("interrupt sparse")
        monkeypatch.setattr(runtime.executor, "sparse", fail)
        with pytest.raises(RuntimeError, match="interrupt sparse"):
            model.sample_action(**inputs)
        assert runtime.last_stats["status"] == "ERROR"
        assert runtime.state.output is None and not runtime.state.kv and not runtime._active
        monkeypatch.setattr(runtime.executor, "sparse", original)
        assert torch.equal(model.sample_action(**inputs), expected)
        with pytest.raises(ValueError, match="num_steps"):
            model.sample_action(**{**inputs, "num_steps": 3})


def test_later_dense_reanchors_and_changed_requests_do_not_leak():
    model, inputs = model_and_inputs()
    conf = config(("dense", "sparse", "dense", "reuse"))
    with HybridVisualRuntime(model, conf) as runtime:
        expected = model.sample_action(**inputs)
        model.sample_action(**{**inputs, "context": inputs["context"] + 2})
        assert torch.equal(model.sample_action(**inputs), expected)
        assert runtime.last_stats["dense_steps"] == 2
    assert not getattr(model, "_visual_ffn_context_cache", None)
