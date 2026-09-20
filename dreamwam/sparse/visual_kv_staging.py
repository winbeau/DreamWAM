"""One dispatch factor: stage read-only action K/V once per visual refresh.

The original action-with-video-cache transformer only reads visual K/V. A
request boundary or any dense/partial refresh invalidates the staged cache;
action tokens, context, modulation and all other inputs are still copied on
every call. No values, selection, kernels or sampling decisions change.
This experimental class is benchmark-only; existing policy options are intact.
"""

from __future__ import annotations

import torch
from torch.utils._pytree import tree_flatten

from .visual_cache_graphs import GraphedVisualTokenCache, TensorTreeReplay


class ActionKVReplay(TensorTreeReplay):
    """The caller owns a monotonically increasing read-only K/V generation."""

    def __init__(self, function, request, **kwargs):
        if type(request) is not dict or "video_kv_cache" not in request:
            raise ValueError("action replay requires an explicit visual K/V subtree")
        super().__init__(function, request, **kwargs)
        offset = 0
        self.kv_indices = set()
        for name, value in request.items():
            leaves, _ = tree_flatten(value)
            if name == "video_kv_cache":
                if not leaves or not all(isinstance(leaf, torch.Tensor) for leaf in leaves):
                    raise ValueError("visual K/V must contain only tensor leaves")
                self.kv_indices.update(range(offset, offset + len(leaves)))
            offset += len(leaves)
        self.kv_bytes = sum(self.buffers[i].numel() * self.buffers[i].element_size()
                            for i in self.kv_indices)
        self._generation = None
        self._staged_generation = None
        self.kv_stage_batches = 0
        self.kv_skip_batches = 0

    def _fill(self, request):
        leaves, spec = tree_flatten(request)
        if spec != self.spec or [self._signature(value) for value in leaves] != self.signature:
            raise ValueError("graph input structure, shape, dtype, device, stride or metadata changed")
        refresh = self._staged_generation != self._generation
        for index, (buffer, value) in enumerate(zip(self.buffers, leaves)):
            if isinstance(buffer, torch.Tensor) and (refresh or index not in self.kv_indices):
                buffer.copy_(value)
        self._staged_generation = self._generation
        self.kv_stage_batches += int(refresh)
        self.kv_skip_batches += int(not refresh)

    def __call__(self, request, *, generation):
        if type(generation) is not int or generation < 1:
            raise ValueError("visual K/V generation must be a positive integer")
        if self._generation is not None and generation < self._generation:
            raise ValueError("visual K/V generation cannot move backwards")
        self._generation = generation
        return super().__call__(request)


class StagedVisualKVCache(GraphedVisualTokenCache):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._kv_generation = 0

    def _begin(self):
        super()._begin()
        self._kv_generation += 1
        self._stats.update(action_kv_stage_batches=0, action_kv_skip_batches=0,
                           action_kv_copied_bytes=0)

    def _refresh(self, forward, **kwargs):
        result = super()._refresh(forward, **kwargs)
        # Covers both dense and partial refresh, including the eager partial
        # control. Never infer freshness from tensor addresses or versions.
        self._kv_generation += 1
        return result

    def _call(self, name, function, request):
        if name != "action":
            return super()._call(name, function, request)
        if name not in self.graphs:
            self.graphs[name] = ActionKVReplay(
                function, request, device=next(self.model.parameters()).device,
                enabled=self.graph_enabled, warmup=self.graph_warmup)
        replay = self.graphs[name]
        staged, skipped = replay.kv_stage_batches, replay.kv_skip_batches
        result = replay(request, generation=self._kv_generation)
        self._stats["action_buffered_calls"] += 1
        self._stats["action_graph_replays"] += int(replay.graph is not None)
        self._stats["action_kv_stage_batches"] += replay.kv_stage_batches - staged
        self._stats["action_kv_skip_batches"] += replay.kv_skip_batches - skipped
        self._stats["action_kv_copied_bytes"] += (replay.kv_stage_batches - staged) * replay.kv_bytes
        return result
