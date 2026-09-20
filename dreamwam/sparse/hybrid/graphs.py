"""Bounded dispatch cache around exactly the eager executor's tensor functions."""

from collections import OrderedDict

import torch
from torch.utils._pytree import tree_flatten

from ..visual_cache_graphs import TensorTreeReplay
from .schedule import stable_hash


class ExecutionDispatcher:
    def __init__(self, model, config):
        self.model, self.config = model, config
        self.entries = OrderedDict()
        self.weight_signature = None
        self.last_key = None
        self.last_replayed = False
        if config.backend == "cuda_graph" and next(model.parameters()).device.type != "cuda":
            raise ValueError("cuda_graph requires a CUDA model")

    def begin_request(self):
        # Graphs embed weight addresses; changes to weights/device/dtype or buffers
        # invalidate capture. Current request tensors are filled on EVERY replay.
        signature = tuple((id(t), t._version, t.dtype, t.device, tuple(t.shape))
                          for t in (*self.model.parameters(), *self.model.buffers()))
        if signature != self.weight_signature:
            self.clear()
            self.weight_signature = signature

    def __call__(self, operation, function, request):
        self.last_replayed = False
        self.last_key = None
        if self.config.backend == "eager":
            return function(**request)
        leaves, spec = tree_flatten(request)
        key = stable_hash(dict(operation=operation, structure=str(spec),
                               signature=[str(TensorTreeReplay._signature(v)) for v in leaves]))
        self.last_key = key
        if key not in self.entries:
            if len(self.entries) >= self.config.max_graphs:
                self.entries.popitem(last=False)
            self.entries[key] = (operation, TensorTreeReplay(
                function, request, device=next(self.model.parameters()).device,
                enabled=self.config.backend == "cuda_graph", warmup=self.config.graph_warmup))
        self.entries.move_to_end(key)
        replay = self.entries[key][1]
        result = replay(request)
        self.last_replayed = replay.graph is not None
        return result

    def stats(self):
        return {key: dict(operation=op, captured=replay.graph is not None,
                          capture_seconds=replay.capture_seconds, replays=replay.replays)
                for key, (op, replay) in self.entries.items()}

    def clear(self):
        self.entries.clear()
        self.last_key = None
        self.last_replayed = False
