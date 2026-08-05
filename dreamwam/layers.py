import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    num_heads: int,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    batch, query_length, width = query.shape
    if width % num_heads != 0:
        raise ValueError(f"attention width {width} is not divisible by {num_heads} heads.")
    head_dim = width // num_heads
    query = query.view(batch, query_length, num_heads, head_dim).transpose(1, 2)
    key = key.view(batch, key.shape[1], num_heads, head_dim).transpose(1, 2)
    value = value.view(batch, value.shape[1], num_heads, head_dim).transpose(1, 2)
    if attention_mask is not None and attention_mask.ndim == 3:
        attention_mask = attention_mask.unsqueeze(1)
    output = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
    )
    return output.transpose(1, 2).reshape(batch, query_length, width)


def modulate(
    tokens: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    return tokens * (1.0 + scale) + shift


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    if dim <= 0 or dim % 2 != 0:
        raise ValueError(f"sinusoidal embedding dim must be positive and even, got {dim}.")
    position = position.reshape(-1)
    frequency = torch.pow(
        10000.0,
        -torch.arange(dim // 2, dtype=torch.float64, device=position.device)
        / (dim // 2),
    )
    phase = torch.outer(position.double(), frequency)
    return torch.cat([torch.cos(phase), torch.sin(phase)], dim=1).to(position.dtype)


def precompute_rope(dim: int, length: int = 1024) -> torch.Tensor:
    if dim <= 0 or dim % 2 != 0:
        raise ValueError(f"RoPE dim must be positive and even, got {dim}.")
    frequency = 1.0 / (
        10000.0
        ** (
            torch.arange(0, dim, 2, dtype=torch.float64)[: dim // 2]
            / dim
        )
    )
    phase = torch.outer(torch.arange(length), frequency)
    return torch.polar(torch.ones_like(phase), phase)


def apply_rope(
    tokens: torch.Tensor,
    frequencies: torch.Tensor,
    num_heads: int,
) -> torch.Tensor:
    batch, sequence, width = tokens.shape
    if width % num_heads != 0:
        raise ValueError(f"token width {width} is not divisible by {num_heads} heads.")
    head_dim = width // num_heads
    if head_dim % 2 != 0:
        raise ValueError(f"RoPE head dim must be even, got {head_dim}.")
    tokens = tokens.view(batch, sequence, num_heads, head_dim)
    complex_tokens = torch.view_as_complex(
        tokens.double().reshape(batch, sequence, num_heads, head_dim // 2, 2)
    )
    complex_output = complex_tokens * frequencies.to(
        device=tokens.device,
        dtype=torch.complex128,
    )
    return torch.view_as_real(complex_output).reshape(batch, sequence, width).to(tokens.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1.0e-5):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        dtype = tokens.dtype
        normalized = tokens.float() * torch.rsqrt(
            tokens.float().pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        return normalized.to(dtype) * self.weight


class SelfAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        attn_head_dim: int,
        num_heads: int,
        eps: float,
    ):
        super().__init__()
        self.num_heads = int(num_heads)
        self.attn_width = int(attn_head_dim) * self.num_heads
        self.q = nn.Linear(hidden_dim, self.attn_width)
        self.k = nn.Linear(hidden_dim, self.attn_width)
        self.v = nn.Linear(hidden_dim, self.attn_width)
        self.o = nn.Linear(self.attn_width, hidden_dim)
        self.norm_q = RMSNorm(self.attn_width, eps)
        self.norm_k = RMSNorm(self.attn_width, eps)


class CrossAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        attn_head_dim: int,
        num_heads: int,
        eps: float,
    ):
        super().__init__()
        self.num_heads = int(num_heads)
        self.attn_width = int(attn_head_dim) * self.num_heads
        self.q = nn.Linear(hidden_dim, self.attn_width)
        self.k = nn.Linear(hidden_dim, self.attn_width)
        self.v = nn.Linear(hidden_dim, self.attn_width)
        self.o = nn.Linear(self.attn_width, hidden_dim)
        self.norm_q = RMSNorm(self.attn_width, eps)
        self.norm_k = RMSNorm(self.attn_width, eps)

    def forward(
        self,
        tokens: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        query = self.norm_q(self.q(tokens))
        key = self.norm_k(self.k(context))
        value = self.v(context)
        attended = scaled_dot_product_attention(
            query,
            key,
            value,
            self.num_heads,
            context_mask,
        )
        return self.o(attended)


class DiTBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        attn_head_dim: int,
        num_heads: int,
        ffn_dim: int,
        eps: float,
    ):
        super().__init__()
        self.num_heads = int(num_heads)
        self.self_attn = SelfAttention(hidden_dim, attn_head_dim, num_heads, eps)
        self.cross_attn = CrossAttention(hidden_dim, attn_head_dim, num_heads, eps)
        self.norm1 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(hidden_dim, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.modulation = nn.Parameter(
            torch.randn(1, 6, hidden_dim) / math.sqrt(hidden_dim)
        )


class PatchHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        out_dim: int,
        patch_size: tuple[int, int, int],
        eps: float,
    ):
        super().__init__()
        self.patch_size = tuple(int(value) for value in patch_size)
        self.norm = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.proj = nn.Linear(hidden_dim, out_dim * math.prod(self.patch_size))
        self.modulation = nn.Parameter(
            torch.randn(1, 2, hidden_dim) / math.sqrt(hidden_dim)
        )

    def forward(
        self,
        tokens: torch.Tensor,
        time_embedding: torch.Tensor,
    ) -> torch.Tensor:
        if time_embedding.ndim == 3:
            modulation = self.modulation.unsqueeze(0) + time_embedding.unsqueeze(2)
            shift, scale = modulation.chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        elif time_embedding.ndim == 2:
            shift, scale = (self.modulation + time_embedding.unsqueeze(1)).chunk(
                2,
                dim=1,
            )
        else:
            raise ValueError(
                "time_embedding must have shape [B,D] or [B,S,D], "
                f"got {tuple(time_embedding.shape)}."
            )
        return self.proj(self.norm(tokens) * (1.0 + scale) + shift)
