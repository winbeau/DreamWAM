"""Causal decision/support proxies; no schedule, budget mutation or cache writes.

The online signal is a first-layer probe, NOT a causal importance oracle. Offline
interventions and benchmark-owned success must determine whether it is useful.
"""

from dataclasses import dataclass
import math

import torch

from dreamwam.layers import apply_rope, modulate
from ..action_guided_neuron_cache import joint_video_mass
from ..av import _to_heads
from ..reuse import token_drift
from .execution import selected_video_state


@dataclass(frozen=True)
class TokenScores:
    query: torch.Tensor | None
    read: torch.Tensor | None
    action_probe_rows: int = 0
    video_key_probe_rows: int = 0
    video_query_probe_rows: int = 0


def normalize(value):
    return value / value.mean(dim=-1, keepdim=True).clamp_min(1e-8)


def project(model, state, kind):
    """Project only requested first-layer Q or K, with original time and RoPE."""
    mot = model.mot
    block = mot.video_expert.blocks[0]
    shift, scale, *_ = mot._split_modulation(block, state["time_modulation"])
    current = modulate(block.norm1(state["tokens"]), shift, scale)
    attention = block.self_attn
    value = getattr(attention, "norm_" + kind)(getattr(attention, kind)(current))
    return apply_rope(value, state["freqs"], mot.num_heads)


def support_mass(query, key, weights, *, num_heads, mask):
    """Propagate seed importance BACKWARD to keys that support those queries.

    s[j] = sum_i d[i] P_VV[i,j]. The opposite direction, P @ d, would instead
    rank consumers of important keys. Visibility and per-seed normalization
    follow native V->V attention; no action keys are visible to video queries.
    """
    q, k = _to_heads(query, num_heads), _to_heads(key, num_heads)
    logits = (q * q.shape[-1] ** -0.5) @ k.transpose(-1, -2)
    probabilities = logits.masked_fill(~mask.to(torch.bool), float("-inf")).float().softmax(-1)
    return (probabilities.mean(dim=1) * weights[:, :, None]).sum(dim=1)


def decision_scores(config, model, video_state, action_state, state, *, anchor=False):
    """Separate read value from refresh urgency, using only CURRENT inputs.

    action: A->[V,A] relevance; action_context: relevance + one-hop VV support;
    visual_context: incoming VV mass with uniform seed weights (no action).
    These deliberately pay for current full-video K instead of using stale K.
    """
    current = video_state["tokens"]
    batch, length = current.shape[:2]
    mot = model.mot
    key = project(model, video_state, "k")
    direct = current.new_zeros((batch, length), dtype=torch.float32)
    action_rows = 0
    if config.selection != "visual_context":
        io = mot._attention_input(mot.action_expert.blocks[0], action_state["tokens"],
                                  action_state["freqs"], action_state["time_modulation"])
        direct = joint_video_mass(io[0], key, io[1], num_heads=mot.num_heads,
                                  action_mask=state.full_mask[length:])
        action_rows = batch * action_state["tokens"].shape[1]
    importance = normalize(direct)
    query_rows = 0
    if config.selection == "visual_context" or (config.selection == "action_context" and config.context_weight > 0):
        if config.selection == "visual_context":
            seeds = torch.arange(length, device=current.device)
            weights = torch.ones_like(direct) / length
        else:
            count = max(1, math.ceil(length * config.support_seed_ratio))
            seeds = direct.mean(dim=0).argsort(descending=True, stable=True)[:count]
            weights = direct.index_select(1, seeds)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        query = project(model, selected_video_state(video_state, seeds), "q")
        support = support_mass(query, key, weights, num_heads=mot.num_heads,
                               mask=state.full_mask[:length, :length].index_select(0, seeds))
        importance = (normalize(support) if config.selection == "visual_context" else
                      importance + config.context_weight * normalize(support))
        query_rows = batch * seeds.numel()
    # Important but unchanged rows should remain readable; they need not be
    # recomputed. Legacy action_drift conflated these two ranking objectives.
    urgency = importance if anchor else token_drift(current.float(), state.reference.float()) * (
        1 + config.guidance_weight * normalize(importance))
    return TokenScores(urgency.mean(dim=0), importance.mean(dim=0),
                       action_rows, batch * length, query_rows)
