"""Static-buffer execution, optionally replayed as a CUDA graph.

The dispatch hypothesis says most of a request goes into dispatching tens of thousands of small
operators, in which case replaying the same work as a single graph removes that cost without
changing a single number - the only route on the table that could reach the target with no
quality risk at all. Two independent things have to be true for it to work, and they are kept
separable on purpose:

1. **Buffering correctness** - every input must land in the fixed addresses the graph was
   captured against, and the output must be read back without aliasing the caller's tensors.
   This is pure bookkeeping and is verified here on CPU.
2. **Capture feasibility and speed** - only a GPU can answer it, and the sampler reports the
   failure reason instead of silently running the eager path, because a silent fallback would
   make a benchmark report the baseline as if it were the graph.
"""

from __future__ import annotations

from typing import Callable, Mapping

import torch


class StaticInputBuffers:
    """Fixed-address copies of a call's tensor inputs.

    CUDA graph replay reads and writes the same memory every time, so the caller's tensors
    cannot be used directly: their addresses change between calls.
    """

    def __init__(self, request: Mapping[str, torch.Tensor], device: torch.device):
        self.keys = tuple(request.keys())
        self.buffers: dict[str, torch.Tensor] = {}
        for name, value in request.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"static buffering only supports tensor inputs, got {type(value).__name__} "
                    f"for {name!r}"
                )
            buffer = torch.empty_like(value, device=device)
            buffer.copy_(value)
            self.buffers[name] = buffer

    def fill(self, request: Mapping[str, torch.Tensor]) -> None:
        if set(request) != set(self.keys):
            raise KeyError(
                "request keys changed between calls: "
                f"{sorted(request)} vs {sorted(self.keys)}; a graph cannot change its inputs"
            )
        for name in self.keys:
            value = request[name]
            buffer = self.buffers[name]
            if tuple(value.shape) != tuple(buffer.shape):
                raise ValueError(
                    f"input {name!r} changed shape from {tuple(buffer.shape)} to "
                    f"{tuple(value.shape)}; a graph cannot change its shapes"
                )
            buffer.copy_(value)

    def as_kwargs(self) -> dict[str, torch.Tensor]:
        return dict(self.buffers)


class StaticBufferSampler:
    """Run ``function(**inputs)`` through fixed buffers, as a CUDA graph when possible.

    ``capture()`` is explicit and idempotent.  If it cannot capture, the reason is kept on
    :attr:`capture_error` and the sampler runs the function eagerly - correct, but the caller
    must be able to see that it happened, which is what the attribute is for.
    """

    def __init__(
        self,
        function: Callable[..., torch.Tensor],
        request: Mapping[str, torch.Tensor],
        *,
        device: torch.device | str,
        warmup: int = 3,
    ):
        self.function = function
        self.device = torch.device(device)
        self.warmup = int(warmup)
        self.buffers = StaticInputBuffers(request, self.device)
        self.graph: torch.cuda.CUDAGraph | None = None
        self.capture_error: str | None = None
        self.output: torch.Tensor | None = None

    @property
    def replayed(self) -> bool:
        return self.graph is not None

    def capture(self) -> bool:
        """Try to capture; return whether replay is available."""
        if self.device.type != "cuda":
            self.capture_error = (
                "CUDA graphs need a CUDA device; running the eager path instead"
            )
            return False
        if not torch.cuda.is_available():
            self.capture_error = "torch reports no CUDA device available"
            return False
        try:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(self.warmup):
                    self.function(**self.buffers.as_kwargs())
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self.output = self.function(**self.buffers.as_kwargs())
            graph.replay()
            torch.cuda.synchronize()
            self.graph = graph
            self.capture_error = None
            return True
        except Exception as error:  # noqa: BLE001 - the reason is the useful part
            self.capture_error = f"{type(error).__name__}: {error}"
            self.graph = None
            self.output = None
            return False

    def __call__(self, request: Mapping[str, torch.Tensor]) -> torch.Tensor:
        self.buffers.fill(request)
        if self.graph is not None:
            self.graph.replay()
            assert self.output is not None
            return self.output.clone()
        # Eager fallback: same buffers, same order, just dispatched normally.
        return self.function(**self.buffers.as_kwargs()).clone()
