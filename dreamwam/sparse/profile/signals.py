"""Descriptive proxies on native post-RoPE Q/K and unrotated V, never labels.

The complete joint key denominator is retained for every selected head/query.
These computations are offline and cannot supply future teacher data to a policy.
"""

import torch


def joint_probabilities(query, key, mask):
    """Input [heads, tokens, channels], original boolean [queries, keys] mask."""
    if query.ndim != 3 or key.ndim != 3 or query.shape[0] != key.shape[0] or query.shape[2] != key.shape[2]:
        raise ValueError("expected compatible complete heads")
    if mask.dtype != torch.bool or mask.shape != (query.shape[1], key.shape[1]):
        raise ValueError("expected original joint boolean mask")
    if not mask.any(dim=-1).all():
        raise ValueError("every query must have a visible key")
    if not torch.isfinite(query).all() or not torch.isfinite(key).all():
        raise ValueError("nonfinite Q/K")
    logits = (query.float() @ key.float().transpose(-1, -2)) * query.shape[-1] ** -0.5
    return logits.masked_fill(~mask, -torch.inf).softmax(dim=-1)


def token_signals(probabilities, value, grid):
    """Per-head scores; video-time differences compare spatially aligned cells.

    value_action = mean_action_query(P_AV) * ||V_j||_2. This is NOT the
    unpublished DIDO action V-attribution; it omits cancellation and output
    projection. Context is P_VV^T d, with no renormalization over visual keys.
    """
    nv = grid.length
    if value.ndim != 3 or probabilities.shape != (value.shape[0], value.shape[1], value.shape[1]):
        raise ValueError("probability/value shape mismatch")
    if value.shape[1] <= nv or not torch.isfinite(value).all():
        raise ValueError("require finite values and all action keys")
    av = probabilities[:, nv:, :nv].mean(dim=1)
    vv = probabilities[:, :nv, :nv]
    visual = value[:, :nv].float()
    value_norm = visual.norm(dim=-1)
    frames = visual.reshape(visual.shape[0], grid.frames, grid.frame_size, visual.shape[-1])
    temporal = (frames - frames[:, :1]).norm(dim=-1).flatten(1)
    weights = av / av.sum(dim=-1, keepdim=True).clamp_min(1e-30)
    context = torch.einsum("hij,hi->hj", vv, weights)
    return dict(action=av, value_norm=value_norm, value_action=av * value_norm,
                value_video_time=temporal, visual_context=vv.mean(dim=1),
                action_context_support=context,
                action_key_mass=probabilities[:, nv:, nv:].sum(dim=-1).mean(dim=-1))


def value_denoising_drift(current, previous, video_length):
    """Same layer/head/position across denoising steps; separate from video time."""
    if current.shape != previous.shape:
        raise ValueError("cannot compare incompatible request layouts")
    return (current[:, :video_length].float() - previous[:, :video_length].float()).norm(dim=-1)
