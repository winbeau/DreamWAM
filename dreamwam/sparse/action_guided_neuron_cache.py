"""An isolated action-guidance factor on top of visual neuron caching.

At each existing dense group anchor, sum A->[V,A] softmax mass over action
queries, then average heads. Mean-normalized visual mass weights the same
post-activation neuron contribution used by the unguided cache. The score is
sum_r |h[r,n]| * (1 + guidance_weight * mass[r]) * ||W_down[:,n]||_2.

Only selection changes: the native attention output still comes from the native
kernel. Extracting mass is extra work, included in full-request measurements.
The selected compact weights and mask remain shared over the original group.
No future denoising step or future policy request supplies the anchor signal.
"""

from __future__ import annotations

from functools import wraps
import math

import torch

from .av import _to_heads
from .ffn_neuron_cache import VisualFFNNeuronCache


def joint_video_mass(query_action, key_video, key_action, *, num_heads, action_mask):
    """A->V probability mass with action keys retained in the denominator."""
    video_length = key_video.shape[1]
    query = _to_heads(query_action, num_heads)
    keys = _to_heads(torch.cat((key_video, key_action), dim=1), num_heads)
    logits = (query * (query.shape[-1] ** -0.5)) @ keys.transpose(-1, -2)
    if action_mask is not None:
        logits = logits.masked_fill(~action_mask.to(torch.bool), float("-inf"))
    probabilities = logits.float().softmax(dim=-1)
    return probabilities[..., :video_length].sum(dim=-2).mean(dim=1)


class ActionGuidedFFNNeuronCache(VisualFFNNeuronCache):
    def __init__(self, model, *, keep_ratio, group_size=10, guidance_weight=1.0):
        super().__init__(model, keep_ratio=keep_ratio, group_size=group_size)
        if not math.isfinite(guidance_weight) or guidance_weight < 0:
            raise ValueError("guidance_weight must be finite and non-negative")
        self.guidance_weight = float(guidance_weight)
        self.action_mass = {}

    def _begin(self):
        super()._begin()
        self.action_mass.clear()
        self._stats.update(action_mass_builds=0, guided_mask_builds=0)

    def _end(self):
        super()._end()
        self.action_mass.clear()

    def _neuron_score(self, layer, hidden):
        if self.guidance_weight == 0:
            return super()._neuron_score(layer, hidden)
        mass = self.action_mass.pop(layer)
        if mass.shape != hidden.shape[:2]:
            raise ValueError("action relevance and visual FFN token layout disagree")
        mass = mass / mass.mean(dim=-1, keepdim=True).clamp_min(1e-8)
        weighted = hidden.float().abs() * (1 + self.guidance_weight * mass).unsqueeze(-1)
        self._stats["guided_mask_builds"] += 1
        return weighted.sum(dim=(0, 1)) * self.weight_norms[layer]

    def __enter__(self):
        super().__enter__()
        mot = self.model.mot
        original = mot._joint_self_attention

        @wraps(original)
        def joint_attention(**kwargs):
            sparse = kwargs.get("sparse")
            if sparse is not None and sparse.enabled:
                raise ValueError("action-guidance factor requires native dense attention")
            layer = kwargs["layer_index"]
            anchor = self.step_counts.get(layer, 0) % self.group_size == 0
            if self.guidance_weight > 0 and 0 < self.keep_ratio < 1 and anchor:
                video_io, action_io = kwargs["video_io"], kwargs["action_io"]
                self.action_mass[layer] = joint_video_mass(
                    action_io[0], video_io[1], action_io[1], num_heads=mot.num_heads,
                    action_mask=kwargs["attention_mask"][kwargs["video_length"]:],
                )
                self._stats["action_mass_builds"] += 1
            return original(**kwargs)

        self._originals.append((mot, "_joint_self_attention", original))
        mot._joint_self_attention = joint_attention
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        self.action_mass.clear()
