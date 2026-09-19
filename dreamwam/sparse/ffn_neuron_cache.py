"""Action Cache's neuron-update rule, applied to the Joint visual FFNs.

An online dense anchor selects neurons by (sum |h|) * ||W_down[:, n]||_2.
The selected up/down weights are gathered once per step group. Other neurons
retain their anchor contribution; they are not zeroed or removed from weights.
This is an input-dependent, request-local ablation, not rMuscle's retrieval
library or the action-guided extension. Attention and action FFNs stay dense.
"""

from __future__ import annotations

import math
import time

import torch
from torch.nn import functional as F

from .ffn_context_cache import VisualFFNContextCache


class VisualFFNNeuronCache(VisualFFNContextCache):
    def __init__(self, model, *, keep_ratio: float, group_size: int = 10):
        super().__init__(model, keep_ratio=keep_ratio)
        if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size < 1:
            raise ValueError("group_size must be a positive integer")
        self.group_size = group_size
        self.weight_norms = {}
        self.step_counts = {}
        self.setup_seconds = 0.0

    @torch.no_grad()
    def prepare(self):
        """Static weight norms once at setup; no checkpoint parameter is altered."""
        if self.weight_norms or self.keep_ratio in (0.0, 1.0):
            return
        device = next(self.model.parameters()).device
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for layer, block in enumerate(self.model.video_expert.blocks):
            if len(block.ffn) != 3:
                raise ValueError("expected Linear/activation/Linear visual FFN")
            # PyTorch Linear stores [output, input], hence a neuron is a COLUMN
            # of the down projection (not a row of its stored tensor).
            self.weight_norms[layer] = block.ffn[2].weight.float().norm(dim=0)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        self.setup_seconds = time.perf_counter() - start

    def _begin(self):
        super()._begin()
        self.step_counts.clear()
        self._stats.update(mask_builds=0, computed_neurons=0, total_neurons=0,
                           gathered_weight_elements=0)

    def _forward(self, layer, original, current):
        if not self._active:
            raise RuntimeError("visual neuron cache must run inside sample_action")
        if current.ndim != 3:
            raise ValueError("visual FFN input must be [batch, tokens, features]")
        block = self.model.video_expert.blocks[layer]
        up, activation, down = block.ffn
        width = up.out_features
        rows = current.shape[0] * current.shape[1]
        self._stats["total_rows"] += rows
        self._stats["computed_rows"] += rows
        self._stats["total_neurons"] += rows * width
        step = self.step_counts.get(layer, 0)
        self.step_counts[layer] = step + 1
        if self.keep_ratio == 1.0:
            self._stats["dense_calls"] += 1
            self._stats["computed_neurons"] += rows * width
            return original(current)
        if step % self.group_size == 0:
            hidden = activation(up(current))
            output = down(hidden)
            self._stats["dense_calls"] += 1
            self._stats["computed_neurons"] += rows * width
            keep = math.ceil(self.keep_ratio * width)
            if keep == 0:
                self.references[layer] = (output, None, None, None, None)
                return output
            if layer not in self.weight_norms:
                raise RuntimeError("call prepare() before measuring the neuron cache")
            score = hidden.float().abs().sum(dim=(0, 1)) * self.weight_norms[layer]
            index = score.argsort(descending=True, stable=True)[:keep]
            selected_up = up.weight.index_select(0, index)
            selected_bias = None if up.bias is None else up.bias.index_select(0, index)
            selected_down = down.weight.index_select(1, index).contiguous()
            anchor_hidden = hidden.index_select(-1, index)
            self.references[layer] = (output, anchor_hidden, selected_up, selected_bias, selected_down)
            self._stats["mask_builds"] += 1
            self._stats["gathered_weight_elements"] += selected_up.numel() + selected_down.numel()
            return output

        anchor_output, anchor_hidden, selected_up, selected_bias, selected_down = self.references[layer]
        if anchor_output.shape != current.shape:
            raise ValueError("visual layout changed within one sampling request")
        self._stats["selective_calls"] += 1
        if anchor_hidden is None:
            return anchor_output
        hidden = activation(F.linear(current, selected_up, selected_bias))
        self._stats["computed_neurons"] += rows * hidden.shape[-1]
        # The bias and the unselected neurons are already in anchor_output.
        # Only a selected-neuron DELTA is added, using the compact weight tensor.
        return anchor_output + F.linear(hidden - anchor_hidden, selected_down)
