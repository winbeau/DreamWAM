"""Action-query attention mass: the A->V anchor signal (M2 input).

The shipped model computes one fused softmax over ``[video; action]`` keys for the action
rows.  Extracting A->V weights therefore has to come from that same normalisation: the
mass an action query places on a video token is only well defined with the action keys in
the denominator.  A video-only renormalisation would be a different, cheaper quantity and
is deliberately not implemented here.

This module returns both the action output and the video mass, so the mass is a by-product
of the action computation rather than a second pass.  It is nevertheless *not* free: the
measured cost of this path versus the fused kernel is recorded in
``docs/implementation/01-dense-profiling.md``.  Callers that do not need the mass should
use ``torch.nn.functional.scaled_dot_product_attention`` directly.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _to_heads(tensor: torch.Tensor, num_heads: int) -> torch.Tensor:
    batch, length, width = tensor.shape
    if width % num_heads:
        raise ValueError(f"width {width} is not divisible by {num_heads} heads")
    return tensor.view(batch, length, num_heads, width // num_heads).transpose(1, 2)


def action_attention_with_mass(
    *,
    query_action: torch.Tensor,
    key_video: torch.Tensor,
    value_video: torch.Tensor,
    key_action: torch.Tensor,
    value_action: torch.Tensor,
    num_heads: int,
    action_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Joint ``A->[V, A]`` attention.

    Returns ``(output, video_mass)`` with ``output`` shaped like ``query_action``
    (``[B, Na, H*D]``) and ``video_mass`` shaped ``[B, H, Nv]``, the total probability
    mass every action query places on each video key.  ``action_mask`` follows the
    library convention ``True == visible`` and applies to the concatenated key axis.
    """
    batch, action_length, width = query_action.shape
    video_length = key_video.shape[1]
    head_dim = width // num_heads

    query = _to_heads(query_action, num_heads) * (head_dim**-0.5)
    key_v = _to_heads(key_video, num_heads)
    key_a = _to_heads(key_action, num_heads)
    value_v = _to_heads(value_video, num_heads)
    value_a = _to_heads(value_action, num_heads)

    logits = torch.cat(
        [query @ key_v.transpose(-1, -2), query @ key_a.transpose(-1, -2)],
        dim=-1,
    )
    if action_mask is not None:
        if action_mask.shape[-1] != video_length + action_length:
            raise ValueError(
                "action_mask must cover the concatenated key axis, got "
                f"{tuple(action_mask.shape)} for {video_length}+{action_length} keys"
            )
        logits = logits.masked_fill(~action_mask.to(torch.bool), float("-inf"))

    # fp32 softmax mirrors the fused kernel's accumulation without changing the result.
    probabilities = torch.softmax(logits.float(), dim=-1).to(logits.dtype)
    video_mass = probabilities[..., :video_length].sum(dim=-2)

    output = probabilities[..., :video_length] @ value_v + probabilities[
        ..., video_length:
    ] @ value_a
    return output.transpose(1, 2).reshape(batch, action_length, width), video_mass
