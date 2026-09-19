"""Token layout: frames, key blocks and the native video visibility rule.

DreamWAM flattens the video latent grid ``(frames, height, width)`` in time-major,
row-major order, so a token index is ``frame * tokens_per_frame + row * width + column``.
The two cameras are concatenated horizontally, which means a "row" spans the seam between
agentview and wrist; block metadata therefore keeps the real grid coordinates instead of
treating flat neighbours as spatial neighbours.

The native visibility mask is *not* causal: frame 0 may only see frame 0, while every
other frame may see the whole video sequence.  Any route produced here is an intersection
with that rule, never a replacement for it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TokenLayout:
    """Partition of the video sequence used by routing and legality checks."""

    video_length: int
    tokens_per_frame: int
    num_frames: int
    block_size: int
    #: Indices of the conditioning frame's keys, ``[tokens_per_frame]``.
    first_frame_keys: torch.Tensor
    #: Future key blocks, ``[num_blocks, block_size]``, padded with ``-1``.
    future_block_keys: torch.Tensor

    @property
    def num_first_frame(self) -> int:
        return int(self.first_frame_keys.numel())

    @property
    def num_future_blocks(self) -> int:
        return int(self.future_block_keys.shape[0])

    @property
    def num_future_keys(self) -> int:
        return self.video_length - self.num_first_frame

    def frame_of(self, device: torch.device | None = None) -> torch.Tensor:
        target = self.first_frame_keys.device if device is None else device
        index = torch.arange(self.video_length, device=target)
        return index // self.tokens_per_frame

    def describe(self) -> dict:
        return {
            "video_length": self.video_length,
            "tokens_per_frame": self.tokens_per_frame,
            "num_frames": self.num_frames,
            "block_size": self.block_size,
            "num_first_frame": self.num_first_frame,
            "num_future_blocks": self.num_future_blocks,
            "num_future_keys": self.num_future_keys,
        }


def build_token_layout(
    *,
    video_length: int,
    tokens_per_frame: int,
    block_size: int,
    device: torch.device,
) -> TokenLayout:
    """Partition ``video_length`` tokens into a conditioning frame and future blocks."""
    if tokens_per_frame <= 0:
        raise ValueError(f"tokens_per_frame must be positive, got {tokens_per_frame}")
    if video_length <= 0 or video_length % tokens_per_frame:
        raise ValueError(
            f"video_length {video_length} must be a positive multiple of "
            f"tokens_per_frame {tokens_per_frame}"
        )
    num_frames = video_length // tokens_per_frame
    if num_frames < 2:
        raise ValueError(
            "sparse video attention needs a conditioning frame plus at least one "
            f"future frame, got {num_frames} frame(s)"
        )
    if block_size < 1:
        raise ValueError(f"block_size must be positive, got {block_size}")

    first_frame_keys = torch.arange(tokens_per_frame, device=device, dtype=torch.long)
    future = torch.arange(
        tokens_per_frame, video_length, device=device, dtype=torch.long
    )
    num_blocks = (future.numel() + block_size - 1) // block_size
    padding = num_blocks * block_size - future.numel()
    if padding:
        future = torch.cat(
            [future, torch.full((padding,), -1, device=device, dtype=torch.long)]
        )
    return TokenLayout(
        video_length=video_length,
        tokens_per_frame=tokens_per_frame,
        num_frames=num_frames,
        block_size=block_size,
        first_frame_keys=first_frame_keys,
        future_block_keys=future.view(num_blocks, block_size),
    )


def video_legality(layout: TokenLayout) -> torch.Tensor:
    """Native ``[video_length, video_length]`` visibility, ``True`` meaning visible."""
    frame_of = layout.frame_of()
    query_is_future = frame_of != 0
    key_is_first_frame = frame_of == 0
    return query_is_future.unsqueeze(1) | key_is_first_frame.unsqueeze(0)


def block_hit_mask(layout: TokenLayout, block_ids: torch.Tensor) -> torch.Tensor:
    """``[B, H, video_length]`` membership of the selected future blocks.

    ``block_ids`` is ``[B, H, m]`` and may contain ``-1`` for "no selection"; the
    conditioning frame is *not* included here, callers add it explicitly.
    """
    if block_ids.ndim != 3:
        raise ValueError(f"block_ids must be [B,H,m], got {tuple(block_ids.shape)}")
    batch, heads, _ = block_ids.shape
    saver = block_ids.clamp(min=0).reshape(batch, heads, -1)
    valid = (block_ids >= 0).reshape(batch, heads, -1)
    tokens = layout.future_block_keys[saver]  # [B, H, m, block_size]
    keep = valid.unsqueeze(-1).expand_as(tokens)
    tokens = torch.where(keep, tokens, torch.full_like(tokens, -1)).reshape(
        batch, heads, -1
    )
    # Same OR-through-a-scratch-column trick as the route membership: padding is clamped to
    # index 0, so writing its False directly would erase the real token 0.
    membership = torch.zeros(
        batch,
        heads,
        layout.video_length + 1,
        dtype=torch.bool,
        device=layout.first_frame_keys.device,
    )
    membership.scatter_(
        2,
        torch.where(tokens >= 0, tokens.clamp(min=0), layout.video_length),
        tokens >= 0,
    )
    return membership[..., : layout.video_length]
