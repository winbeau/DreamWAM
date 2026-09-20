"""GPU-native proxies from real dense-layer projections, with no teacher inputs.

These operations also execute inside the dense CUDA graph and its measured cost.
Scores are refreshed only by Dense steps; retaining scores is structural state,
never permission to inject old K/V in structure-only execution.
"""

import torch


def scoring_heads(total, limit):
    count = min(total, limit)
    return tuple(i * total // count for i in range(count))


def score_work(config, *, layers, num_heads, video_length, action_length):
    wanted = ({name for name, weight in config.weights if weight > 0} if config.signal == "fusion" else {config.signal})
    multiplier = layers * min(num_heads, config.heads)
    queries = (action_length if wanted & {"action", "value_action", "action_context"} else 0)
    queries += video_length if wanted & {"visual_context", "action_context"} else 0
    return dict(native_score_query_head_rows=multiplier * queries,
                native_score_value_head_rows=multiplier * video_length if wanted & {"value_norm", "value_action", "dynamic"} else 0)


def to_heads(value, num_heads, ids):
    batch, length, width = value.shape
    return value.reshape(batch, length, num_heads, width // num_heads).permute(0, 2, 1, 3)[:, ids].float()


def native_layer_scores(config, video_io, action_io, *, num_heads, frame_size, mask):
    nv = video_io[0].shape[1]
    if nv % frame_size:
        raise ValueError("native score requires complete frames")
    ids = scoring_heads(num_heads, config.heads)
    if config.signal == "uniform":
        return video_io[0].new_zeros(nv, dtype=torch.float32)
    wanted = ({name for name, weight in config.weights if weight > 0} if config.signal == "fusion" else {config.signal})
    values = {}
    need_action = bool(wanted & {"action", "value_action", "action_context"})
    need_vv = bool(wanted & {"visual_context", "action_context"})
    need_values = bool(wanted & {"value_norm", "value_action", "dynamic"})
    av = value = None
    if need_action or need_vv:
        key = to_heads(torch.cat((video_io[1], action_io[1]), dim=1), num_heads, ids)
        scale = key.shape[-1] ** -0.5
    if need_action:
        query = to_heads(action_io[0], num_heads, ids)
        logits = (query @ key.transpose(-1, -2)) * scale
        probabilities = logits.masked_fill(~mask[nv:], -torch.inf).softmax(dim=-1)
        av = probabilities[..., :nv].mean(dim=-2)
        values["action"] = av
    if need_values:
        value = to_heads(video_io[2], num_heads, ids)
        values["value_norm"] = value.norm(dim=-1)
        if "value_action" in wanted:
            values["value_action"] = av * values["value_norm"]
        if "dynamic" in wanted:
            frames = value.reshape(*value.shape[:2], nv // frame_size, frame_size, value.shape[-1])
            values["dynamic"] = (frames - frames[:, :, :1]).norm(dim=-1).flatten(2)
    if need_vv:
        query = to_heads(video_io[0], num_heads, ids)
        logits = (query @ key.transpose(-1, -2)) * scale
        probabilities = logits.masked_fill(~mask[:nv], -torch.inf).softmax(dim=-1)[..., :nv]
        if "visual_context" in wanted:
            values["visual_context"] = probabilities.mean(dim=-2)
        if "action_context" in wanted:
            weights = av / av.sum(dim=-1, keepdim=True).clamp_min(1e-30)
            values["action_context"] = torch.einsum("bhij,bhi->bhj", probabilities, weights)
    reduced = {name: value.mean(dim=(0, 1)) for name, value in values.items() if name in wanted}
    if config.signal != "fusion":
        return reduced[config.signal]
    # Weights are explicit development-study parameters, never learned here.
    return sum(weight * reduced[name] / reduced[name].mean().clamp_min(1e-30)
               for name, weight in config.weights if weight > 0)


def safe_depth_scores(scores):
    valid = torch.isfinite(scores).all(dim=-1) & (scores >= 0).all(dim=-1)
    safe = torch.where(valid[:, None], scores, torch.zeros_like(scores))
    # Equal influence for explicitly sampled depths despite different activation scales.
    normalized = safe / safe.mean(dim=-1, keepdim=True).clamp_min(1e-30)
    return normalized, valid


def read_quotas(native, count, frame_size, frames):
    if native.observed == "full":
        if count < frame_size or (frames > 1 and count - frame_size < frames - 1):
            raise ValueError("observed-full budget must retain the observed frame and cover future frames")
        if frames == 1:
            if count != frame_size:
                raise ValueError("one observed frame requires its full budget")
            return [frame_size]
        remaining = count - frame_size
        quotas = [frame_size] + [remaining // (frames - 1) + int(f < remaining % (frames - 1))
                                 for f in range(frames - 1)]
    else:
        quotas = [count // frames + int(f < count % frames) for f in range(frames)]
    if any(not 1 <= quota <= frame_size for quota in quotas):
        raise ValueError("native read quota exceeds a frame or leaves it empty")
    return quotas


def choose_native(score, count, length, device, *, valid):
    uniform = torch.arange(count, device=device) * length // max(count, 1)
    if score is None:
        return uniform
    usable = valid & torch.isfinite(score).all()
    safe = torch.nan_to_num(score, nan=0, posinf=0, neginf=0)
    ranked = safe.argsort(descending=True, stable=True)[:count]
    return torch.where(usable, ranked, uniform)


def native_read_route(config, score, valid, *, count, frame_size, frames):
    result = []
    for frame, quota in enumerate(read_quotas(config, count, frame_size, frames)):
        start = frame * frame_size
        local = None if config.signal == "uniform" or (frame == 0 and config.observed == "uniform") else score[start:start + frame_size]
        result.append(choose_native(local, quota, frame_size, score.device, valid=valid) + start)
    return torch.cat(result).sort().values
