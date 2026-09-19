"""Request-local selective recomputation of the Joint visual expert's FFNs.

This is a Context Cache ablation, not rMuscle's cross-execution cache library.
The first denoising step supplies a dense reference. Later steps rank visual
rows by relative FFN-input drift and update only the selected rows. Each cached
input remains paired with the output actually computed from that input.

The context manager changes only visual FFN calls and restores all methods on
exit. No parameter, action FFN, attention, sampler, or world residual is changed.
Request boundaries (including exceptional exits) discard all tensor references.
"""

from __future__ import annotations

import math
from functools import wraps

import torch

from .reuse import merge_reused_output, token_drift


class VisualFFNContextCache:
    def __init__(self, model, *, keep_ratio: float):
        if not math.isfinite(keep_ratio) or not 0.0 <= keep_ratio <= 1.0:
            raise ValueError("keep_ratio must be finite and in [0, 1]")
        if model.config.setting != "joint":
            raise ValueError("visual FFN context cache requires the Joint model")
        if model.training:
            raise ValueError("visual FFN context cache is inference-only")
        self.model = model
        self.keep_ratio = float(keep_ratio)
        self.references = {}
        self.last_stats = {}
        self._stats = {}
        self._active = False
        self._originals = []

    def _begin(self):
        if self._active:
            raise RuntimeError("concurrent/reentrant use of one model is unsupported")
        self.references.clear()
        self._stats = dict(dense_calls=0, selective_calls=0, computed_rows=0, total_rows=0)
        self._active = True

    def _end(self):
        self.last_stats = dict(self._stats)
        self.references.clear()
        self._active = False

    def _forward(self, layer, original, current):
        if not self._active:
            raise RuntimeError("visual FFN cache must run inside sample_action")
        if current.ndim != 3:
            raise ValueError("visual FFN input must be [batch, tokens, features]")
        rows = current.shape[0] * current.shape[1]
        self._stats["total_rows"] += rows
        reference = self.references.get(layer)
        if self.keep_ratio == 1.0 or reference is None:
            result = original(current)
            self._stats["dense_calls"] += 1
            self._stats["computed_rows"] += rows
            if self.keep_ratio < 1.0:
                self.references[layer] = (current.clone(), result.clone())
            return result

        previous_input, previous_output = reference
        if previous_input.shape != current.shape:
            raise ValueError("visual layout changed within one sampling request")
        keep = math.ceil(self.keep_ratio * current.shape[1])
        self._stats["selective_calls"] += 1
        self._stats["computed_rows"] += current.shape[0] * keep
        if keep == 0:
            return previous_output
        drift = token_drift(current, previous_input)
        index = drift.argsort(dim=-1, descending=True, stable=True)[..., :keep]
        gather_index = index.unsqueeze(-1).expand(-1, -1, current.shape[-1])
        selected_input = current.gather(1, gather_index)
        selected_output = original(selected_input)
        result = merge_reused_output(previous_output, selected_output, index)
        # Updating the entire reference input here would underestimate accumulated
        # drift on reused rows. Only rows whose FFNs ran advance their reference.
        previous_input.scatter_(1, gather_index, selected_input)
        self.references[layer] = (previous_input, result)
        return result

    def __enter__(self):
        if self._originals or getattr(self.model, "_visual_ffn_context_cache", None):
            raise RuntimeError("a visual FFN context cache is already installed")
        self.model._visual_ffn_context_cache = self
        for layer, block in enumerate(self.model.video_expert.blocks):
            original = block.ffn.forward

            @wraps(original)
            def forward(current, _layer=layer, _original=original):
                return self._forward(_layer, _original, current)

            self._originals.append((block.ffn, "forward", original))
            block.ffn.forward = forward
        sample = self.model.sample_action

        @wraps(sample)
        def sample_action(*args, **kwargs):
            self._begin()
            try:
                return sample(*args, **kwargs)
            finally:
                self._end()

        self._originals.append((self.model, "sample_action", sample))
        self.model.sample_action = sample_action
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for owner, name, original in reversed(self._originals):
            setattr(owner, name, original)
        self._originals.clear()
        self.references.clear()
        self._active = False
        del self.model._visual_ffn_context_cache
