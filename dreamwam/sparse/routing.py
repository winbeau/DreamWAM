"""M2: turn action anchors and visual statistics into an executable VV route.

The route is *per head and per batch element*: :attr:`Route.keys` holds the video key
indices each head keeps.  It always starts with the whole conditioning frame (frame 0),
which is the structural floor - frame-0 queries may only see frame-0 keys under the native
mask, so dropping any of them would leave those rows without a single legal key.

Selection policies differ in what they are allowed to know, which is what makes them usable
as the paper's ablations:

* ``recency`` / ``uniform`` - no action information at all (visual-only baseline),
* ``av``                    - action anchors only,
* ``av_context``            - anchors re-ranked with a cheap query/key affinity term,
* ``all``                   - full-budget control.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import SparseConfig
from .layout import TokenLayout


@dataclass
class Route:
    """Executable per-head video-key route."""

    keys: torch.Tensor  # [B, H, Kmax] int64, gathered key indices (padding clamped to 0)
    valid: torch.Tensor  # [B, H, Kmax] bool, False marks padding
    membership: torch.Tensor  # [B, H, Nv] bool, selected keys including frame 0
    blocks: torch.Tensor  # [B, H] int, kept future blocks per head
    fallback: torch.Tensor  # [B, H] bool, low anchor confidence fallback fired
    concentration: torch.Tensor  # [B, H] float, top-block share of the action video mass
    density: float  # kept / legal video query-key pairs, averaged over heads
    selection: str
    stage: int

    def describe(self) -> dict:
        return {
            "selection": self.selection,
            "stage": self.stage,
            "kmax": int(self.keys.shape[-1]),
            "blocks_mean": float(self.blocks.float().mean()),
            "blocks_min": int(self.blocks.min().item()) if self.blocks.numel() else 0,
            "blocks_max": int(self.blocks.max().item()) if self.blocks.numel() else 0,
            "fallback_fraction": (
                float(self.fallback.float().mean()) if self.fallback.numel() else 0.0
            ),
            "concentration_mean": (
                float(self.concentration.mean()) if self.concentration.numel() else 0.0
            ),
            "density": self.density,
        }


def _block_token_scores(token_scores: torch.Tensor, layout: TokenLayout) -> torch.Tensor:
    """Sum per-token scores into future-block scores, ``[B,H,Nv] -> [B,H,NB]``."""
    blocks = layout.future_block_keys.clamp(min=0)  # [NB, bs]
    valid = (layout.future_block_keys >= 0).to(token_scores.dtype)  # [NB, bs]
    gathered = token_scores[:, :, blocks]  # [B, H, NB, bs]
    return (gathered * valid).sum(dim=-1)


def _affinity_scores(
    *,
    query_video: torch.Tensor,
    key_video: torch.Tensor,
    layout: TokenLayout,
    num_heads: int,
) -> torch.Tensor:
    """Cheap block-level VV affinity, ``[B, H, NB]``.

    Uses the mean post-RoPE query over the future frames against the mean key of each key
    block.  This is a ranking proxy, not an attention weight: averaging rotated vectors can
    cancel, which is exactly why it only re-ranks anchors instead of defining the route.
    """
    batch, length, width = query_video.shape
    head_dim = width // num_heads
    query = query_video.view(batch, length, num_heads, head_dim).transpose(1, 2)
    key = key_video.view(batch, length, num_heads, head_dim).transpose(1, 2)
    future_queries = query[:, :, layout.num_first_frame :, :].mean(dim=2)  # [B,H,D]

    blocks = layout.future_block_keys.clamp(min=0)  # [NB, bs]
    valid = (layout.future_block_keys >= 0).to(key.dtype)  # [NB, bs]
    block_keys = key[:, :, blocks, :]  # [B,H,NB,bs,D]
    block_keys = (block_keys * valid.view(1, 1, *valid.shape, 1)).sum(dim=3)
    block_keys = block_keys / valid.sum(dim=1).clamp(min=1).view(1, 1, -1, 1)
    return (future_queries.unsqueeze(2) * block_keys).sum(dim=-1) / head_dim**0.5


def _normalize(scores: torch.Tensor) -> torch.Tensor:
    minimum = scores.amin(dim=-1, keepdim=True)
    maximum = scores.amax(dim=-1, keepdim=True)
    return (scores - minimum) / (maximum - minimum).clamp(min=1e-6)


def _ordering(
    *,
    kind: str,
    scores: torch.Tensor | None,
    layout: TokenLayout,
    batch: int,
    num_heads: int,
) -> torch.Tensor:
    """Descending-preference block ordering, ``[B, H, NB]``."""
    num_blocks = layout.num_future_blocks
    device = layout.first_frame_keys.device
    if kind in {"av", "av_context"}:
        if scores is None:
            raise ValueError(f"selection {kind!r} needs anchor scores")
        return torch.argsort(scores, dim=-1, descending=True, stable=True)
    if kind == "recency":
        order = torch.arange(num_blocks - 1, -1, -1, device=device)
    elif kind == "uniform":
        if num_blocks == 1:
            order = torch.zeros(1, dtype=torch.long, device=device)
        else:
            order = torch.round(
                torch.linspace(0, num_blocks - 1, num_blocks, device=device)
            ).long()
    elif kind == "all":
        order = torch.arange(num_blocks, device=device)
    else:
        raise ValueError(f"unknown selection {kind!r}")
    return order.view(1, 1, -1).expand(batch, num_heads, -1)


def build_route(
    *,
    layout: TokenLayout,
    config: SparseConfig,
    num_heads: int,
    step_index: int,
    num_steps: int,
    av_mass: torch.Tensor | None = None,
    query_video: torch.Tensor | None = None,
    key_video: torch.Tensor | None = None,
) -> Route:
    """Build the route executed by one layer at one denoising step."""
    if not config.enabled:
        raise ValueError("build_route called with sparse disabled")
    stage = config.stage_of(step_index, num_steps)
    # Keep the Python tuple alongside the tensor: the widest budget is then known without
    # reading the device, and an .item() here would synchronise once per layer-step.
    blocks_per_head = config.blocks_for(
        num_heads=num_heads,
        stage=stage,
        num_future_blocks=layout.num_future_blocks,
    )
    keep = torch.tensor(
        blocks_per_head,
        device=layout.first_frame_keys.device,
        dtype=torch.long,
    )
    batch = av_mass.shape[0] if av_mass is not None else 1
    device = layout.first_frame_keys.device

    scores: torch.Tensor | None = None
    block_mass: torch.Tensor | None = None
    if config.selection in {"av", "av_context"}:
        if av_mass is None:
            raise ValueError(f"selection {config.selection!r} requires av_mass")
        block_mass = _block_token_scores(av_mass, layout)
        scores = block_mass
        if config.selection == "av_context":
            if query_video is None or key_video is None:
                raise ValueError("selection 'av_context' requires query_video/key_video")
            affinity = _affinity_scores(
                query_video=query_video,
                key_video=key_video,
                layout=layout,
                num_heads=num_heads,
            )
            scores = _normalize(block_mass) + config.context_weight * _normalize(
                affinity
            )

    # Conditioning-frame relevance, available only when the selection policy already needed the
    # anchor mass.  Without it the frame is compressed by position, which is the honest
    # fallback: this is a budget knob, not a claim about which visual tokens matter.
    cond_scores = (
        None if av_mass is None else av_mass[..., : layout.num_first_frame]
    )

    order = _ordering(
        kind=config.selection,
        scores=scores,
        layout=layout,
        batch=batch,
        num_heads=num_heads,
    )
    fallback_order = _ordering(
        kind=config.fallback,
        scores=None,
        layout=layout,
        batch=batch,
        num_heads=num_heads,
    )

    if block_mass is not None:
        total = av_mass.sum(dim=-1).clamp(min=1e-9)
        concentration = block_mass.amax(dim=-1) / total
    else:
        concentration = torch.ones(batch, num_heads, device=device)
    fallback = concentration < config.min_anchor_mass
    # Always blend through torch.where. Guarding this with `bool(fallback.any())` reads a
    # device tensor into Python, which forces a synchronisation and makes the whole denoising
    # loop uncapturable by CUDA graphs - the very overhead this project is trying to remove.
    # The where is unconditional because it is correct either way.
    selected = torch.where(fallback.unsqueeze(-1), fallback_order, order)

    # `keep` comes from a tuple of Python ints, so the maximum is known without touching the
    # device; an .item() here would synchronise once per layer-step.
    maximum = max(blocks_per_head) if blocks_per_head else 0
    # `rank` counts blocks, `keep` counts per head: compare on different axes.
    rank = torch.arange(maximum, device=device).view(1, 1, -1)
    budget = keep.view(1, -1, 1)
    if maximum > 0:
        chosen = selected[:, :, :maximum]
        chosen = torch.where(
            rank < budget, chosen, torch.full_like(chosen, -1)
        )
    else:
        chosen = torch.full((batch, num_heads, 0), -1, device=device, dtype=torch.long)

    block_tokens = layout.future_block_keys[chosen.clamp(min=0)]  # [B,H,m,bs]
    token_valid = chosen.ge(0).unsqueeze(-1) & (
        layout.future_block_keys[chosen.clamp(min=0)] >= 0
    )
    block_tokens = torch.where(
        token_valid, block_tokens, torch.full_like(block_tokens, -1)
    ).reshape(batch, num_heads, -1)

    # Conditioning-frame keys kept for future-frame queries.  The full frame is the default;
    # reducing it is the knob that lets the video branch go below the old 42.86% floor.
    first_keep = int(round(config.conditional_keep_ratio * layout.num_first_frame))
    first_keep = max(1, min(first_keep, layout.num_first_frame))
    first_frame = layout.first_frame_keys.view(1, 1, -1).expand(batch, num_heads, -1)
    if first_keep < layout.num_first_frame:
        # Rank the conditioning frame with the same signal the future blocks use when one is
        # available, so a compressed conditioning frame keeps what the action looks at.
        if cond_scores is not None:
            order = torch.argsort(cond_scores, dim=-1, descending=True, stable=True)
        else:
            order = torch.arange(layout.num_first_frame, device=device).view(1, 1, -1)
            order = order.expand(batch, num_heads, -1)
        first_frame = first_frame.gather(2, order[:, :, :first_keep])
    keys = torch.cat([first_frame, block_tokens], dim=-1)
    valid = keys >= 0
    keys = keys.clamp(min=0)

    # Scatter membership with OR semantics through a scratch column.  Padding entries carry
    # index -1 and are clamped to 0 before scattering; writing them as False directly into the
    # real table erases the legitimate key 0 whenever a head has fewer blocks than the widest
    # head - which is exactly the per-head-budget case M1 produces.  The scratch column absorbs
    # every invalid entry so no real position can be cleared by padding.
    membership = torch.zeros(
        batch, num_heads, layout.video_length + 1, dtype=torch.bool, device=device
    )
    membership.scatter_(2, torch.where(valid, keys, layout.video_length), valid)
    membership = membership[..., : layout.video_length]

    blocks = keep.view(1, -1).expand(batch, num_heads).clone()
    kept_keys = (
        first_keep + blocks.to(torch.float32) * layout.block_size
    )
    future_queries = layout.video_length - layout.num_first_frame
    # Frame-0 queries keep the whole conditioning frame to themselves, so their pairs are
    # unaffected by conditional_keep_ratio; only the future rows shrink.
    kept_pairs = layout.num_first_frame**2 + future_queries * kept_keys
    legal_pairs = layout.num_first_frame**2 + future_queries * layout.video_length
    density = float((kept_pairs / legal_pairs).mean())

    return Route(
        keys=keys,
        valid=valid,
        membership=membership,
        blocks=blocks,
        fallback=fallback,
        concentration=concentration,
        density=density,
        selection=config.selection,
        stage=stage,
    )
