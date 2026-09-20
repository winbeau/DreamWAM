"""Causal token scoring and fixed-budget sets, independent of step scheduling."""

from dataclasses import dataclass

import torch

from ..action_guided_neuron_cache import joint_video_mass
from ..reuse import token_drift


@dataclass(frozen=True)
class Selection:
    query: torch.Tensor
    route: torch.Tensor
    action_probe_rows: int = 0


def scores(config, model, video_state, action_state, state, *, anchor=False):
    current = video_state["tokens"]
    drift = (torch.zeros(current.shape[:2], device=current.device) if anchor else
             token_drift(current.float(), state.reference.float()))
    if config.selection != "action_drift" or config.guidance_weight == 0:
        return (None if config.selection == "uniform" or anchor else drift.mean(dim=0)), 0
    mot = model.mot
    io = mot._attention_input(mot.action_expert.blocks[0], action_state["tokens"],
                              action_state["freqs"], action_state["time_modulation"])
    length = current.shape[1]
    relevance = joint_video_mass(io[0], state.kv[0]["k"], io[1], num_heads=mot.num_heads,
                                action_mask=state.full_mask[length:])
    relevance = relevance / relevance.mean(dim=-1, keepdim=True).clamp_min(1e-8)
    score = relevance if anchor else drift * (1 + config.guidance_weight * relevance)
    return score.mean(dim=0), action_state["tokens"].shape[0] * action_state["tokens"].shape[1]


def choose(score, count, length, device):
    if score is None:
        return torch.arange(count, device=device) * length // max(count, 1)
    return score.argsort(descending=True, stable=True)[:count]


def select(config, model, video_state, action_state, state, *, anchor=False):
    current = video_state["tokens"]
    length = current.shape[1]
    q_count, _ = config.budgets(length, video_state["tokens_per_frame"])
    route = torch.arange(length, device=current.device)
    if anchor:
        return Selection(route, route)
    score, probes = scores(config, model, video_state, action_state, state)
    query = choose(score, q_count, length, current.device).sort().values
    return Selection(query, route, probes)
