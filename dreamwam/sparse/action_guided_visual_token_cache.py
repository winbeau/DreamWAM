"""Current-action guidance as one added factor on selective visual refresh.

At a partial refresh only, read first-layer action Q/K from the CURRENT action
state and its timestep. Joint A->[V,A] normalization over cached visual keys gives
one causal relevance signal. Weight each token's accumulated input drift by
1 + guidance_weight * mean-normalized action relevance. Zero drift stays zero,
so action relevance alone never forces recomputation of unchanged inputs.

This proxy is shared across visual layers. The extra action Q/K/V projection and
mass extraction are counted and timed, and do not advance the action scheduler.
There is no future-step or future-request information and no SR claim here.
"""

from __future__ import annotations

import math

from .action_guided_neuron_cache import joint_video_mass
from .reuse import token_drift
from .visual_token_cache import VisualTokenRefreshCache


class ActionGuidedVisualTokenCache(VisualTokenRefreshCache):
    def __init__(self, model, *, keep_ratio, refresh_every=5, guidance_weight=1.0):
        super().__init__(model, keep_ratio=keep_ratio, refresh_every=refresh_every)
        if not math.isfinite(guidance_weight) or guidance_weight < 0:
            raise ValueError("guidance_weight must be finite and non-negative")
        self.guidance_weight = float(guidance_weight)
        self.action_mass = None

    def _begin(self):
        super()._begin()
        self.action_mass = None
        self._stats.update(action_mass_builds=0, guidance_action_qkv_builds=0)

    def _end(self):
        super()._end()
        self.action_mass = None

    def _refresh(self, forward, **kwargs):
        if self.video_output is not None and 0 < self.keep_ratio < 1 and self.guidance_weight > 0:
            mot = self.model.mot
            action_state, video_state = kwargs["action_state"], kwargs["video_state"]
            action_io = mot._attention_input(
                mot.action_expert.blocks[0], action_state["tokens"],
                action_state["freqs"], action_state["time_modulation"],
            )
            length = video_state["tokens"].shape[1]
            mask = mot.build_attention_mask(
                video_length=length, action_length=action_state["tokens"].shape[1],
                video_tokens_per_frame=video_state["tokens_per_frame"],
                device=action_state["tokens"].device,
            )[length:]
            self.action_mass = joint_video_mass(
                action_io[0], self.video_kv[0]["k"], action_io[1],
                num_heads=mot.num_heads, action_mask=mask,
            )
            self._stats["action_mass_builds"] += 1
            self._stats["guidance_action_qkv_builds"] += 1
        try:
            return super()._refresh(forward, **kwargs)
        finally:
            self.action_mass = None

    def _select(self, current, keep):
        if self.guidance_weight == 0:
            return super()._select(current, keep)
        if self.action_mass is None or self.action_mass.shape != current.shape[:2]:
            raise ValueError("current action relevance is missing or has a different visual layout")
        drift = token_drift(current.float(), self.reference_input.float())
        relevance = self.action_mass / self.action_mass.mean(dim=-1, keepdim=True).clamp_min(1e-8)
        score = (drift * (1 + self.guidance_weight * relevance)).mean(dim=0)
        return score.argsort(descending=True, stable=True)[:keep].sort().values

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        self.action_mass = None
