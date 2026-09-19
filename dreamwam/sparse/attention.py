"""M3: execute the routed VV attention while keeping A->V and A->A exact.

Two backends:

``masked``
    Keeps the shipped full-length attention call for the video rows and restricts the mask
    to the routed keys.  Mathematically identical to the dense call over the routed key set
    and *numerically* the cleanest reference, but it buys no speed: the kernel still walks
    every key.  Measured on the released geometry, ``gather`` (below) is 28% slower than the
    shipped fused call and ``masked`` is at best neutral, so ``masked`` is the default and
    the honest label for it is "no speedup, used to measure the quality effect".

``gather``
    Selects the routed keys into a shorter tensor so the attention kernel really does less
    work.  On DreamWAM's 294-token video sequence the extra launches and the copy cost more
    than the skipped keys, so this backend exists to document the measurement rather than to
    claim a speedup.

Neither backend touches the action rows' joint softmax over ``[video; action]`` keys, and
all hidden states, projections and FFN updates stay dense.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .av import action_attention_with_mass
from .config import SparseConfig
from .layout import TokenLayout, video_legality
from .routing import Route, build_route

NEEDS_ANCHORS = ("av", "av_context")


def _heads(tensor: torch.Tensor, num_heads: int) -> torch.Tensor:
    batch, length, width = tensor.shape
    return tensor.view(batch, length, num_heads, width // num_heads).transpose(1, 2)


def sparse_joint_attention(
    *,
    query_video: torch.Tensor,
    key_video: torch.Tensor,
    value_video: torch.Tensor,
    query_action: torch.Tensor,
    key_action: torch.Tensor,
    value_action: torch.Tensor,
    num_heads: int,
    layout: TokenLayout,
    config: SparseConfig,
    step_index: int,
    num_steps: int,
    action_mask: torch.Tensor | None = None,
    route: Route | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any], Route]:
    """Return ``(video_output, action_output, stats, route)`` for one layer.

    ``route`` may be supplied by the caller to amortize anchor extraction and routing over
    several layers; the returned route is the one actually executed, so the caller can cache
    it.  When a route is supplied the action branch falls back to the fused kernel, because
    the action-anchor signal it would produce is no longer needed - and that extraction is
    the single most expensive part of the routed path.
    """
    if not config.enabled:
        raise ValueError("sparse_joint_attention called with sparse disabled")

    reused = route is not None
    # Anchor extraction reuses the action rows' own computation, but only for the layer that
    # actually builds the route; later layers reuse the cached route and skip it entirely.
    if config.selection in NEEDS_ANCHORS and not reused:
        action_output, av_mass = action_attention_with_mass(
            query_action=query_action,
            key_video=key_video,
            value_video=value_video,
            key_action=key_action,
            value_action=value_action,
            num_heads=num_heads,
            action_mask=action_mask,
        )
    else:
        av_mass = None
        action_output = F.scaled_dot_product_attention(
            _heads(query_action, num_heads),
            _heads(torch.cat([key_video, key_action], dim=1), num_heads),
            _heads(torch.cat([value_video, value_action], dim=1), num_heads),
            attn_mask=action_mask,
        ).transpose(1, 2).reshape(query_action.shape)

    if route is None:
        route = build_route(
            layout=layout,
            config=config,
            num_heads=num_heads,
            step_index=step_index,
            num_steps=num_steps,
            av_mass=av_mass,
            query_video=query_video,
            key_video=key_video,
        )

    if config.backend == "masked":
        # Frame-0 queries always keep dense access to their own frame.  The native mask already
        # forbids them from seeing anything else, and the gather backend computes those rows
        # densely, so applying the routed membership to them here would make the two backends
        # disagree about the same configuration.
        first = layout.num_first_frame
        legality = video_legality(layout)
        mask = (route.membership.unsqueeze(2) & legality.view(
            1, 1, layout.video_length, layout.video_length
        ))
        mask = mask.clone()
        mask[:, :, :first, :first] = True
        video_output = F.scaled_dot_product_attention(
            _heads(query_video, num_heads),
            _heads(key_video, num_heads),
            _heads(value_video, num_heads),
            attn_mask=mask,
        ).transpose(1, 2).reshape(query_video.shape)
    elif config.backend == "gather":
        first = layout.num_first_frame
        query = _heads(query_video, num_heads)
        key = _heads(key_video, num_heads)
        value = _heads(value_video, num_heads)
        # Frame-0 queries may only see frame-0 keys, so their rows are exactly dense over
        # the conditioning frame and need no mask at all.
        head_output = F.scaled_dot_product_attention(
            query[:, :, :first], key[:, :, :first], value[:, :, :first]
        )
        index = route.keys.view(
            route.keys.shape[0], route.keys.shape[1], route.keys.shape[2], 1
        ).expand(-1, -1, -1, key.shape[-1])
        gathered_key = key.gather(2, index)
        gathered_value = value.gather(2, index)
        tail_output = F.scaled_dot_product_attention(
            query[:, :, first:],
            gathered_key,
            gathered_value,
            attn_mask=route.valid.unsqueeze(2),
        )
        video_output = torch.cat([head_output, tail_output], dim=2).transpose(
            1, 2
        ).reshape(query_video.shape)
    else:
        raise ValueError(f"unknown backend {config.backend!r}")

    stats = route.describe()
    stats["backend"] = config.backend
    stats["av_mass_mean"] = float(av_mass.mean()) if av_mass is not None else None
    stats["anchors_recomputed"] = not reused
    return video_output, action_output, stats, route
