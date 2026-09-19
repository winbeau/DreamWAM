from typing import Any, Optional

import torch
from torch import nn

from .sparse.fastops import rope_tables
from .layers import (
    DiTBlock,
    PatchHead,
    precompute_rope,
    sinusoidal_embedding_1d,
)


class VideoDiT(nn.Module):
    def __init__(
        self,
        *,
        video_latent_dim: int,
        flow_latent_dim: int,
        hidden_dim: int,
        ffn_dim: int,
        text_dim: int,
        freq_dim: int,
        patch_size: tuple[int, int, int],
        num_heads: int,
        attn_head_dim: int,
        num_layers: int,
        eps: float,
    ):
        super().__init__()
        self.video_latent_dim = int(video_latent_dim)
        self.flow_latent_dim = int(flow_latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.freq_dim = int(freq_dim)
        self.patch_size = tuple(int(value) for value in patch_size)
        self.num_heads = int(num_heads)
        self.attn_head_dim = int(attn_head_dim)
        self.num_layers = int(num_layers)
        if self.patch_size[0] != 1:
            raise ValueError("DreamWAM uses temporal patch size 1.")
        if min(self.video_latent_dim, self.flow_latent_dim, self.num_layers) <= 0:
            raise ValueError("Video, flow, and layer dimensions must be positive.")

        joint_latent_dim = self.video_latent_dim + self.flow_latent_dim
        self.patch_embedding = nn.Conv3d(
            joint_latent_dim,
            hidden_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 6),
        )
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_dim=hidden_dim,
                    attn_head_dim=attn_head_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    eps=eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.head = PatchHead(
            hidden_dim=hidden_dim,
            out_dim=joint_latent_dim,
            patch_size=self.patch_size,
            eps=eps,
        )

        frame_dim = attn_head_dim - 2 * (attn_head_dim // 3)
        spatial_dim = attn_head_dim // 3
        # Keep complex RoPE frequencies out of Module.to(dtype=...) casts.
        self.frame_freqs = precompute_rope(frame_dim)
        self.height_freqs = precompute_rope(spatial_dim)
        self.width_freqs = precompute_rope(spatial_dim)
        #: Whether ``pre_dit`` builds real (cos, sin) tables instead of complex ones.  Off by
        #: default so the shipped numerics are untouched; :meth:`enable_fast_ops` turns it on.
        self._rope_real = False
        self.frame_cos: torch.Tensor | None = None
        self.frame_sin: torch.Tensor | None = None
        self.height_cos: torch.Tensor | None = None
        self.height_sin: torch.Tensor | None = None
        self.width_cos: torch.Tensor | None = None
        self.width_sin: torch.Tensor | None = None

    def enable_fast_ops(self, *, dtype: torch.dtype = torch.float32) -> None:
        """Build real RoPE tables so rotation avoids float64 and complex128.

        Explicit and per-instance on purpose: a global switch is how a run ends up mislabelled
        about what it executed.  Both ``pre_dit`` and the shared ``apply_rope`` accept either
        representation, so calling this changes only the cost, not the computation.
        """
        frame_dim = self.attn_head_dim - 2 * (self.attn_head_dim // 3)
        spatial_dim = self.attn_head_dim // 3
        self.frame_cos, self.frame_sin = rope_tables(frame_dim, dtype=dtype)
        self.height_cos, self.height_sin = rope_tables(spatial_dim, dtype=dtype)
        self.width_cos, self.width_sin = rope_tables(spatial_dim, dtype=dtype)
        self._rope_real = True

    @property
    def rope_is_real(self) -> bool:
        return self._rope_real

    def _validate_inputs(
        self,
        video_latents: torch.Tensor,
        flow_latents: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> None:
        if video_latents.ndim != 5:
            raise ValueError(
                "video_latents must be [B,C,T,H,W], "
                f"got {tuple(video_latents.shape)}."
            )
        if flow_latents.ndim != 5:
            raise ValueError(
                f"flow_latents must be [B,C,T,H,W], got {tuple(flow_latents.shape)}."
            )
        if video_latents.shape[1] != self.video_latent_dim:
            raise ValueError(
                f"expected {self.video_latent_dim} video channels, "
                f"got {video_latents.shape[1]}."
            )
        if flow_latents.shape[1] != self.flow_latent_dim:
            raise ValueError(
                f"expected {self.flow_latent_dim} flow channels, "
                f"got {flow_latents.shape[1]}."
            )
        if video_latents.shape[0] != flow_latents.shape[0] or (
            video_latents.shape[2:] != flow_latents.shape[2:]
        ):
            raise ValueError(
                "Video and flow must align in batch, time, and space: "
                f"{tuple(video_latents.shape)} vs {tuple(flow_latents.shape)}."
            )
        if timestep.shape != (video_latents.shape[0],):
            raise ValueError(
                f"timestep must be [B], got {tuple(timestep.shape)}."
            )
        if context.ndim != 3 or context.shape[0] != video_latents.shape[0]:
            raise ValueError(
                f"context must be [B,L,D], got {tuple(context.shape)}."
            )
        if context_mask.shape != context.shape[:2]:
            raise ValueError(
                "context_mask must match context [B,L], "
                f"got {tuple(context_mask.shape)} and {tuple(context.shape)}."
            )
        patch_h, patch_w = self.patch_size[1:]
        if video_latents.shape[3] % patch_h or video_latents.shape[4] % patch_w:
            raise ValueError(
                "Latent spatial size must be divisible by patch size, "
                f"got {tuple(video_latents.shape[3:])} and {self.patch_size[1:]}."
            )

    def pre_dit(
        self,
        *,
        video_latents: torch.Tensor,
        flow_latents: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> dict[str, Any]:
        self._validate_inputs(
            video_latents,
            flow_latents,
            timestep,
            context,
            context_mask,
        )

        # Flow enters VideoDiT only through channel concatenation.
        joint_latents = torch.cat([video_latents, flow_latents], dim=1)
        patches = self.patch_embedding(joint_latents)
        batch, _, frames, height, width = patches.shape
        tokens_per_frame = height * width

        token_timesteps = timestep.view(batch, 1, 1).expand(
            batch,
            frames,
            tokens_per_frame,
        ).clone()
        token_timesteps[:, 0] = 0
        time_embedding = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, token_timesteps.reshape(-1))
        ).reshape(batch, frames * tokens_per_frame, self.hidden_dim)
        time_modulation = self.time_projection(time_embedding).unflatten(
            2,
            (6, self.hidden_dim),
        )

        tokens = patches.permute(0, 2, 3, 4, 1).reshape(
            batch,
            frames * height * width,
            self.hidden_dim,
        )
        context = self.text_embedding(context)
        context_mask = context_mask.to(dtype=torch.bool).unsqueeze(1).expand(
            batch,
            tokens.shape[1],
            context.shape[1],
        )

        if frames > self.frame_freqs.shape[0]:
            raise ValueError(f"frame token grid exceeds RoPE cache: {frames}.")
        if height > self.height_freqs.shape[0] or width > self.width_freqs.shape[0]:
            raise ValueError(f"spatial token grid exceeds RoPE cache: {(height, width)}.")

        def expand(table: torch.Tensor, axis: int, size: int) -> torch.Tensor:
            shape = [1, 1, 1, -1]
            shape[axis] = size
            return table[:size].view(shape).expand(frames, height, width, -1)

        if self._rope_real:
            frame_cos, height_cos, width_cos = self.frame_cos, self.height_cos, self.width_cos
            frame_sin, height_sin, width_sin = self.frame_sin, self.height_sin, self.width_sin
            frequencies = (
                torch.cat(
                    [
                        expand(frame_cos, 0, frames),
                        expand(height_cos, 1, height),
                        expand(width_cos, 2, width),
                    ],
                    dim=-1,
                ).reshape(frames * height * width, 1, -1),
                torch.cat(
                    [
                        expand(frame_sin, 0, frames),
                        expand(height_sin, 1, height),
                        expand(width_sin, 2, width),
                    ],
                    dim=-1,
                ).reshape(frames * height * width, 1, -1),
            )
        else:
            frequencies = torch.cat(
                [
                    self.frame_freqs[:frames]
                    .view(frames, 1, 1, -1)
                    .expand(frames, height, width, -1),
                    self.height_freqs[:height]
                    .view(1, height, 1, -1)
                    .expand(frames, height, width, -1),
                    self.width_freqs[:width]
                    .view(1, 1, width, -1)
                    .expand(frames, height, width, -1),
                ],
                dim=-1,
            ).reshape(frames * height * width, 1, -1)

        return {
            "tokens": tokens,
            "freqs": frequencies,
            "time_embedding": time_embedding,
            "time_modulation": time_modulation,
            "context": context,
            "context_mask": context_mask,
            "grid_size": (frames, height, width),
            "tokens_per_frame": tokens_per_frame,
        }

    def build_attention_mask(
        self,
        sequence_length: int,
        tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        if sequence_length <= 0 or sequence_length % tokens_per_frame:
            raise ValueError(
                "Video sequence length must be a positive multiple of tokens_per_frame."
            )
        mask = torch.ones(
            sequence_length,
            sequence_length,
            dtype=torch.bool,
            device=device,
        )
        mask[:tokens_per_frame, tokens_per_frame:] = False
        return mask

    def unpatchify(
        self,
        patches: torch.Tensor,
        grid_size: tuple[int, int, int],
        channels: int,
    ) -> torch.Tensor:
        batch, sequence, patch_width = patches.shape
        frames, height, width = grid_size
        patch_t, patch_h, patch_w = self.patch_size
        expected_width = patch_t * patch_h * patch_w * channels
        if sequence != frames * height * width or patch_width != expected_width:
            raise ValueError(
                "Patch output shape does not match grid/channels: "
                f"{tuple(patches.shape)}, grid={grid_size}, channels={channels}."
            )
        patches = patches.view(
            batch,
            frames,
            height,
            width,
            patch_t,
            patch_h,
            patch_w,
            channels,
        )
        return (
            patches.permute(0, 7, 1, 4, 2, 5, 3, 6)
            .reshape(
                batch,
                channels,
                frames * patch_t,
                height * patch_h,
                width * patch_w,
            )
            .contiguous()
        )

    def post_dit(
        self,
        tokens: torch.Tensor,
        state: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        joint_channels = self.video_latent_dim + self.flow_latent_dim
        patches = self.head(tokens, state["time_embedding"])
        prediction = self.unpatchify(
            patches,
            state["grid_size"],
            joint_channels,
        )
        return torch.split(
            prediction,
            [self.video_latent_dim, self.flow_latent_dim],
            dim=1,
        )


class ActionDiT(nn.Module):
    def __init__(
        self,
        *,
        action_dim: int,
        hidden_dim: int,
        ffn_dim: int,
        text_dim: int,
        freq_dim: int,
        num_heads: int,
        attn_head_dim: int,
        num_layers: int,
        eps: float,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.freq_dim = int(freq_dim)
        self.num_heads = int(num_heads)
        self.attn_head_dim = int(attn_head_dim)
        self.num_layers = int(num_layers)
        self.action_embedding = nn.Linear(action_dim, hidden_dim)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 6),
        )
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_dim=hidden_dim,
                    attn_head_dim=attn_head_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    eps=eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.head = nn.Linear(hidden_dim, action_dim)
        self.freqs = precompute_rope(attn_head_dim)
        self._rope_real = False
        self.cos: torch.Tensor | None = None
        self.sin: torch.Tensor | None = None

    def enable_fast_ops(self, *, dtype: torch.dtype = torch.float32) -> None:
        """Build real RoPE tables so rotation avoids float64 and complex128."""
        self.cos, self.sin = rope_tables(self.attn_head_dim, dtype=dtype)
        self._rope_real = True

    @property
    def rope_is_real(self) -> bool:
        return self._rope_real

    def pre_dit(
        self,
        *,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> dict[str, Any]:
        if action_tokens.ndim != 3 or action_tokens.shape[2] != self.action_dim:
            raise ValueError(
                f"action_tokens must be [B,T,{self.action_dim}], "
                f"got {tuple(action_tokens.shape)}."
            )
        batch, sequence, _ = action_tokens.shape
        if timestep.shape != (batch,):
            raise ValueError(f"timestep must be [B], got {tuple(timestep.shape)}.")
        if context.ndim != 3 or context.shape[0] != batch:
            raise ValueError(f"context must be [B,L,D], got {tuple(context.shape)}.")
        if context_mask.shape != context.shape[:2]:
            raise ValueError(
                f"context_mask must be [B,L], got {tuple(context_mask.shape)}."
            )
        if sequence > self.freqs.shape[0]:
            raise ValueError(f"action sequence exceeds RoPE cache: {sequence}.")

        time_embedding = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep)
        )
        time_modulation = self.time_projection(time_embedding).unflatten(
            1,
            (6, self.hidden_dim),
        )
        context = self.text_embedding(context)
        context_mask = context_mask.to(dtype=torch.bool).unsqueeze(1).expand(
            batch,
            sequence,
            context.shape[1],
        )
        return {
            "tokens": self.action_embedding(action_tokens),
            "freqs": (
                (self.cos[:sequence], self.sin[:sequence])
                if self._rope_real
                else self.freqs[:sequence].view(sequence, 1, -1)
            ),
            "time_modulation": time_modulation,
            "context": context,
            "context_mask": context_mask,
        }

    def post_dit(
        self,
        tokens: torch.Tensor,
        state: Optional[dict[str, Any]] = None,
    ) -> torch.Tensor:
        del state
        return self.head(tokens)
