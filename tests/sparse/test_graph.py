"""Buffering correctness for static-buffer execution.

These run on CPU, which is the point: capture feasibility needs a GPU, but whether the inputs
land in the right place and the output is read back safely does not, so that half is settled
here instead of consuming a GPU window. The eager fallback is exercised on every run, so the
fallback path cannot rot unnoticed.
"""

from __future__ import annotations

import pytest
import torch

from dreamwam.sparse.graph import StaticBufferSampler, StaticInputBuffers


def doubling(**inputs):
    return inputs["x"] * 2 + inputs["y"]


def make_request(value: float = 1.0) -> dict:
    return {"x": torch.full((2, 3), value), "y": torch.full((2, 3), value)}


# --- buffers ----------------------------------------------------------------------


def test_buffers_do_not_alias_the_caller():
    """A graph writes into its own memory, so the caller's tensor must stay untouched."""
    request = make_request()
    buffers = StaticInputBuffers(request, torch.device("cpu"))
    request["x"].add_(100.0)
    assert torch.allclose(buffers.buffers["x"], torch.ones(2, 3)), (
        "the buffer kept a reference instead of a copy"
    )


def test_fill_rejects_a_changed_key_set():
    """A graph cannot change its inputs; silently accepting this would run stale data."""
    buffers = StaticInputBuffers(make_request(), torch.device("cpu"))
    with pytest.raises(KeyError, match="keys changed"):
        buffers.fill({"x": torch.ones(2, 3)})


def test_fill_rejects_a_changed_shape():
    buffers = StaticInputBuffers(make_request(), torch.device("cpu"))
    with pytest.raises(ValueError, match="changed shape"):
        buffers.fill({"x": torch.ones(4, 3), "y": torch.ones(2, 3)})


def test_fill_rejects_non_tensor_inputs():
    with pytest.raises(TypeError, match="tensor inputs"):
        StaticInputBuffers({"x": torch.ones(1), "n": 3}, torch.device("cpu"))


# --- sampler ----------------------------------------------------------------------


def test_sampler_matches_a_direct_call():
    sampler = StaticBufferSampler(doubling, make_request(1.0), device="cpu")
    assert torch.allclose(sampler(make_request(1.0)), doubling(**make_request(1.0)))


def test_sampler_tracks_new_inputs_between_calls():
    """Two calls with different inputs must give different answers - no stale buffer."""
    sampler = StaticBufferSampler(doubling, make_request(1.0), device="cpu")
    first = sampler(make_request(1.0))
    second = sampler(make_request(5.0))
    assert not torch.allclose(first, second)
    assert torch.allclose(second, doubling(**make_request(5.0)))


def test_sampler_does_not_return_the_internal_buffer():
    """Returning the buffer itself would let the next call overwrite the caller's result."""
    sampler = StaticBufferSampler(doubling, make_request(1.0), device="cpu")
    first = sampler(make_request(1.0))
    sampler(make_request(9.0))
    assert torch.allclose(first, torch.full((2, 3), 3.0)), (
        "the returned tensor shares memory with the sampler's output buffer"
    )


def test_cpu_sampler_reports_why_it_is_not_replaying():
    """A silent fallback would let a benchmark report the baseline as if it were the graph."""
    sampler = StaticBufferSampler(doubling, make_request(), device="cpu")
    assert sampler.capture() is False
    assert sampler.replayed is False
    assert sampler.capture_error and "CUDA" in sampler.capture_error


def test_capture_is_idempotent_and_reports_success_consistently():
    sampler = StaticBufferSampler(doubling, make_request(), device="cpu")
    first = sampler.capture()
    error = sampler.capture_error
    second = sampler.capture()
    assert first == second is False
    assert sampler.capture_error == error
