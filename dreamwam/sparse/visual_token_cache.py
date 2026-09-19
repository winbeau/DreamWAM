"""Selective whole-transformer token refresh at a fixed temporal cadence.

The first refresh is dense. Later refreshes gather visual queries by accumulated
input drift and run their projections, attention, FFNs and world residuals. Each
layer updates only those K/V rows. Attention retains every visual and action key
under the original visibility mask. Unselected final video rows retain their last
computed values, keeping the original grid for video prediction and scheduling.

This extends token reuse beyond FFNs. It is request-local and causal, and is an
approximation requiring official paired SR. This first factor has no action
guidance. Temporal cadence is held fixed when measuring the token budget.
"""

from __future__ import annotations

import math

import torch

from dreamwam.layers import scaled_dot_product_attention
from .reuse import token_drift
from .visual_step_cache import VisualStepCache


class VisualTokenRefreshCache(VisualStepCache):
    def __init__(self, model, *, keep_ratio, refresh_every=5):
        super().__init__(model, refresh_every=refresh_every)
        if not math.isfinite(keep_ratio) or not 0 <= keep_ratio <= 1:
            raise ValueError("keep_ratio must be finite and in [0, 1]")
        self.keep_ratio = float(keep_ratio)
        self.reference_input = None
        self._indices = []
        self._layout = None

    def _begin(self):
        super()._begin()
        self.reference_input = None
        self._indices.clear()
        self._layout = None
        self._stats.update(partial_video_steps=0, mask_builds=0,
                           computed_video_token_layers=0)

    def _end(self):
        if self._layout is not None:
            batch, length = self._layout
            self._stats["total_video_token_layers"] = batch * length * self._stats["action_layer_updates"]
        self._stats["selected_indices"] = [index.cpu().tolist() for index in self._indices]
        super()._end()
        self.reference_input = None
        self._indices.clear()
        self._layout = None

    def _select(self, current, keep):
        # A single shared mask preserves one RoPE position sequence across the batch.
        # Each input reference advances only when that row is actually recomputed.
        drift = token_drift(current.float(), self.reference_input.float()).mean(dim=0)
        return drift.argsort(descending=True, stable=True)[:keep].sort().values

    def _refresh(self, forward, **kwargs):
        video_state, action_state = kwargs["video_state"], kwargs["action_state"]
        current = video_state["tokens"]
        batch, length, _ = current.shape
        mot = self.model.mot
        self._layout = (batch, length)
        if self.video_output is None or self.keep_ratio == 1:
            result = super()._refresh(forward, **kwargs)
            if self.keep_ratio < 1:
                self.reference_input = current.clone()
            self._stats["computed_video_token_layers"] += batch * length * mot.num_layers
            return result
        if self.reference_input.shape != current.shape:
            raise ValueError("visual layout changed within a request")
        keep = math.ceil(length * self.keep_ratio)
        if keep == 0:
            return self._reuse(video_state=video_state, action_state=action_state)
        index = self._select(current, keep)
        selected_input = current.index_select(1, index)
        self._indices.append(index)
        self._stats["mask_builds"] += 1
        action, video = self._partial_refresh(
            video_state, action_state, kwargs["residual_injection"], index, selected_input,
        )
        self.reference_input.index_copy_(1, index, selected_input)
        self.video_output = self.video_output.index_copy(1, index, video)
        self._stats["partial_video_steps"] += 1
        self._stats["video_layer_updates"] += mot.num_layers
        self._stats["computed_video_token_layers"] += batch * keep * mot.num_layers
        return {"video": self.video_output, "action": action}

    def _partial_refresh(self, video_state, action_state, residual_injection, index, video):
        mot = self.model.mot
        length = video_state["tokens"].shape[1]
        keep = index.numel()
        action = action_state["tokens"]
        full_mask = mot.build_attention_mask(
            video_length=length, action_length=action.shape[1],
            video_tokens_per_frame=video_state["tokens_per_frame"], device=action.device,
        )
        # Query rows are gathered; key columns remain in their original positions.
        mask = torch.cat((full_mask.index_select(0, index), full_mask[length:]), dim=0)
        freqs = video_state["freqs"]
        freqs = (tuple(part.index_select(0, index) for part in freqs)
                 if isinstance(freqs, tuple) else freqs.index_select(0, index))
        modulation = video_state["time_modulation"]
        if modulation.ndim == 4:
            modulation = modulation.index_select(1, index)
        context_mask = video_state["context_mask"]
        if context_mask.ndim == 3:
            context_mask = context_mask.index_select(1, index)
        for layer in range(mot.num_layers):
            video_block, action_block = mot.video_expert.blocks[layer], mot.action_expert.blocks[layer]
            video_io = mot._attention_input(video_block, video, freqs, modulation)
            action_io = mot._attention_input(
                action_block, action, action_state["freqs"], action_state["time_modulation"],
            )
            cache = self.video_kv[layer]
            cache["k"].index_copy_(1, index, video_io[1])
            cache["v"].index_copy_(1, index, video_io[2])
            mixed = scaled_dot_product_attention(
                torch.cat((video_io[0], action_io[0]), dim=1),
                torch.cat((cache["k"], action_io[1]), dim=1),
                torch.cat((cache["v"], action_io[2]), dim=1), mot.num_heads, mask,
            )
            video = mot._post_attention(
                video_block, video_io[3], mixed[:, :keep], *video_io[4:],
                video_state["context"], context_mask,
            )
            action = mot._post_attention(
                action_block, action_io[3], mixed[:, keep:], *action_io[4:],
                action_state["context"], action_state["context_mask"],
            )
            if residual_injection is not None:
                video = residual_injection(layer, video, video_state["context"], context_mask)
        return action, video

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        self.reference_input = None
        self._indices.clear()
        self._layout = None
