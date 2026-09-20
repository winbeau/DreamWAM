"""Causal token scoring and fixed-budget sets, independent of step scheduling."""

from dataclasses import dataclass

import torch

from ..action_guided_neuron_cache import joint_video_mass
from ..reuse import token_drift
from .routing import TokenScores, decision_scores


@dataclass(frozen=True)
class Selection:
    query: torch.Tensor
    route: torch.Tensor
    action_probe_rows: int = 0
    video_key_probe_rows: int = 0
    video_query_probe_rows: int = 0


def scores(config, model, video_state, action_state, state, *, anchor=False):
    if config.selection in ("action", "action_context", "visual_context"):
        return decision_scores(config, model, video_state, action_state, state, anchor=anchor)
    current = video_state["tokens"]
    drift = (torch.zeros(current.shape[:2], device=current.device) if anchor else
             token_drift(current.float(), state.reference.float()))
    if config.selection != "action_drift" or config.guidance_weight == 0:
        score = None if config.selection == "uniform" or anchor else drift.mean(dim=0)
        return TokenScores(score, score)
    mot = model.mot
    io = mot._attention_input(mot.action_expert.blocks[0], action_state["tokens"],
                              action_state["freqs"], action_state["time_modulation"])
    length = current.shape[1]
    relevance = joint_video_mass(io[0], state.kv[0]["k"], io[1], num_heads=mot.num_heads,
                                action_mask=state.full_mask[length:])
    relevance = relevance / relevance.mean(dim=-1, keepdim=True).clamp_min(1e-8)
    score = relevance if anchor else drift * (1 + config.guidance_weight * relevance)
    return TokenScores(score.mean(dim=0), score.mean(dim=0),
                       action_state["tokens"].shape[0] * action_state["tokens"].shape[1])


def choose(score, count, length, device):
    if score is None:
        return torch.arange(count, device=device) * length // max(count, 1)
    return score.argsort(descending=True, stable=True)[:count]


def select(config, model, video_state, action_state, state, *, anchor=False):
    current = video_state["tokens"]
    length = current.shape[1]
    frame_size = video_state["tokens_per_frame"]
    q_count, kv_count = config.budgets(length, frame_size)
    route = torch.arange(length, device=current.device)
    if anchor and config.read_mode == "full":
        return Selection(route, route)
    ranking = scores(config, model, video_state, action_state, state, anchor=anchor)
    score, read_score = ranking.query, ranking.read
    probes = (ranking.action_probe_rows, ranking.video_key_probe_rows, ranking.video_query_probe_rows)
    if config.read_mode == "compact":
        frames = length // frame_size
        queries, routes = [], []
        offset = 0
        for frame in range(frames):
            q = q_count // frames + int(frame < q_count % frames)
            k = kv_count // frames + int(frame < kv_count % frames)
            start = frame * frame_size
            local_score = None if score is None else score[start:start + frame_size]
            if anchor:
                selected = choose(None if read_score is None else read_score[start:start + frame_size],
                                  k, frame_size, current.device) + start
            else:
                query = choose(local_score, q, frame_size, current.device) + start
                queries.append(query)
                old = state.route[offset:offset + k]
                priority = -old.float() if read_score is None else read_score.index_select(0, old)
                already_selected = (old[:, None] == query[None, :]).any(dim=1)
                priority = priority.masked_fill(already_selected, float("-inf"))
                remaining = old[priority.argsort(descending=True, stable=True)[:k - q]]
                selected = torch.cat((query, remaining))
            routes.append(selected)
            offset += k
        # Each newly read key is a freshly computed query; old keys fill the
        # remaining per-frame quota. Cardinalities never depend on tensor values.
        query = route if anchor else torch.cat(queries).sort().values
        return Selection(query, torch.cat(routes).sort().values, *probes)
    if config.frame_quota == "balanced":
        frames = length // frame_size
        query = torch.cat([
            choose(None if score is None else score[frame * frame_size:(frame + 1) * frame_size],
                   q_count // frames + int(frame < q_count % frames), frame_size, current.device)
            + frame * frame_size for frame in range(frames)
        ]).sort().values
    else:
        query = choose(score, q_count, length, current.device).sort().values
    return Selection(query, route, *probes)
