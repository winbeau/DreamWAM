"""Graph/buffered execution cannot change schedules or retain old input values."""

from dataclasses import replace

import pytest
import torch

from dreamwam.sparse.hybrid.runtime import HybridVisualRuntime
from test_hybrid_compact import compact
from test_visual_step_cache import model_and_inputs


@pytest.mark.parametrize("backend", ["buffered", "cuda_graph"])
@pytest.mark.parametrize("read_mode", ["full", "compact"])
def test_dispatch_parity_after_changing_inputs_and_poisoning_buffers(backend, read_mode):
    if backend == "cuda_graph" and not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    device = "cuda" if backend == "cuda_graph" else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    with torch.device(device):
        model, inputs = model_and_inputs()
    model.to(dtype=dtype)
    inputs = {key: value.to(device=device, dtype=dtype if value.is_floating_point() else value.dtype)
              if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}
    conf = replace(compact(selection="action_drift"), read_mode=read_mode,
                   read_ratio=1.0 if read_mode == "full" else 0.5)
    eager = HybridVisualRuntime(model, conf)
    runtime = HybridVisualRuntime(model, replace(conf, backend=backend))
    requests = (inputs, {**inputs, "context": inputs["context"] + 1,
                         "first_frame_latents": inputs["first_frame_latents"] * -2}, inputs)
    for request in requests:
        with eager:
            expected = model.sample_action(**request)
        for _, replay in runtime.dispatch.entries.values():
            for buffer in replay.buffers:
                if isinstance(buffer, torch.Tensor) and buffer.is_floating_point():
                    buffer.fill_(float("nan"))
        with runtime:
            actual = model.sample_action(**request)
            assert torch.equal(actual, expected)
            assert runtime.last_stats["denoising_steps"] == 4
            assert runtime.last_stats["graph_replays"] == (4 if device == "cuda" else 0)
            assert runtime.last_stats["computed_video_token_layers"] == 36
            assert len(runtime.dispatch.entries) == 3
    assert all(row["captured"] == (device == "cuda") for row in runtime.graph_stats().values())
    runtime.close_graphs()
    assert not runtime.graph_stats()


def test_weight_changes_invalidate_graph_objects_and_cache_limit_is_enforced():
    model, inputs = model_and_inputs()
    conf = replace(compact(), backend="buffered", max_graphs=2)
    with HybridVisualRuntime(model, conf) as runtime:
        model.sample_action(**inputs)
        previous = [replay for _, replay in runtime.dispatch.entries.values()]
        assert len(previous) == 2
        with torch.no_grad():
            next(model.parameters()).add_(0.01)
        model.sample_action(**inputs)
        assert not any(replay in previous for _, replay in runtime.dispatch.entries.values())
        assert len(runtime.dispatch.entries) == 2
