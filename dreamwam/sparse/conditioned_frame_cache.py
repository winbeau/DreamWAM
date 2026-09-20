"""Experimental reuse of the invariant conditioned frame at visual refreshes.

The released Joint sampler restores video frame zero, fixes its flow and time,
and masks its queries from future frames and actions. Patch time size is one;
cross-attention and world routing have no inter-video-token reduction. Its K/V
and final hidden rows are therefore mathematically invariant within a request.
All future-frame rows are recomputed at every existing visual refresh. Reduced
GEMM/attention shapes can still change floating-point results: real-checkpoint
action parity must be measured, never inferred from this argument.

The policy exposes this measured factor through conditioned_frame_reuse. The
strengthened Dense control uses refresh_every=1. Graph scope, action steps,
schedulers, precision and RNG stay unchanged.
"""

from __future__ import annotations

import math

import torch

from .visual_cache_graphs import GraphedVisualTokenCache
from .visual_token_cache import VisualTokenRefreshCache


class _ConditionedFrameSelection:
    def _begin(self):
        super()._begin()
        self._conditioned_layout = None
        self._conditioned_gates = {}
        self._conditioned_previews = {}
        self._stats["conditioned_token_layer_reuses"] = 0

    def _end(self):
        super()._end()
        self._conditioned_layout = None
        self._conditioned_gates.clear()
        self._conditioned_previews.clear()

    def _select(self, current, keep):
        prefix, length = self._conditioned_layout
        if keep != length - prefix or current.shape[1] != length:
            raise ValueError("conditioned-frame selection must retain every future query")
        return torch.arange(prefix, length, device=current.device)

    def _refresh(self, forward, **kwargs):
        state = kwargs["video_state"]
        prefix, length = int(state["tokens_per_frame"]), state["tokens"].shape[1]
        if self.model.config.patch_size[0] != 1 or not 0 < prefix < length:
            raise ValueError("conditioned-frame reuse requires temporal patch one and future frames")
        if self._conditioned_layout is not None and self._conditioned_layout != (prefix, length):
            raise ValueError("conditioned frame layout changed inside a request")
        self._conditioned_layout = (prefix, length)
        self.keep_ratio = (length - prefix) / length
        if math.ceil(length * self.keep_ratio) != length - prefix:
            self.keep_ratio = math.nextafter(self.keep_ratio, 0.0)
        anchor = self.video_output is None
        result = super()._refresh(forward, **kwargs)
        router = self.model.world_residual
        if anchor:
            self._conditioned_gates = {key: [value[:, :prefix].clone() for value in values]
                                      for key, values in router._runtime_gates.items()}
            self._conditioned_previews = {key: [(layer, value[:, :prefix].clone()) for layer, value in values]
                                         for key, values in router._runtime_local_preview_tokens.items()}
        else:
            # Restore full diagnostic layouts as well as the full visual output.
            for key, values in router._runtime_gates.items():
                anchors = self._conditioned_gates[key]
                if len(anchors) != len(values):
                    raise RuntimeError("world gate layout changed")
                router._runtime_gates[key] = [torch.cat((saved, current), dim=1)
                                             for saved, current in zip(anchors, values)]
            for key, values in router._runtime_local_preview_tokens.items():
                anchors = self._conditioned_previews[key]
                if [layer for layer, _ in anchors] != [layer for layer, _ in values]:
                    raise RuntimeError("world preview layout changed")
                router._runtime_local_preview_tokens[key] = [(layer, torch.cat((saved, current), dim=1))
                    for (layer, saved), (_, current) in zip(anchors, values)]
            self._stats["conditioned_token_layer_reuses"] += state["tokens"].shape[0] * prefix * self.model.mot.num_layers
        return result


class ConditionedFrameCache(_ConditionedFrameSelection, VisualTokenRefreshCache):
    def __init__(self, model, *, refresh_every=1):
        # The actual fraction is derived from the first request's token layout.
        super().__init__(model, keep_ratio=0.5, refresh_every=refresh_every)


class GraphedConditionedFrameCache(_ConditionedFrameSelection, GraphedVisualTokenCache):
    def __init__(self, model, *, refresh_every=1, graph_enabled=True, graph_warmup=3):
        super().__init__(model, keep_ratio=0.5, refresh_every=refresh_every,
                         guidance_weight=0.0, graph_enabled=graph_enabled,
                         graph_warmup=graph_warmup, graph_partial=True)
