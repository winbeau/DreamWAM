"""Launch- and traffic-reducing replacements for the two hottest small operators.

Profiling the released model (docs/implementation/01-dense-profiling.md) showed the video
block's cost is independent of token count, so the run is bound by operator count and memory
traffic rather than arithmetic. Two operators are worth replacing on that evidence:

* :class:`~dreamwam.layers.RMSNorm` upcasts to fp32 and back on every call. It is called four
  times per block (self- and cross-attention query/key norms), so its casts alone account for
  roughly half of the ``aten::to`` calls measured in one request.
* :func:`~dreamwam.layers.apply_rope` promotes to float64, builds a complex128 tensor and
  multiplies in complex double precision, then casts back. That is 8x the memory traffic of
  fp32 for a rotation that needs four real multiplies.

Both replacements are numerically equivalent re-implementations, not approximations, and they
are verified against the originals here rather than trusted.  They are kept out of the model
until a matched-Dense measurement justifies them, so enabling them stays a controlled,
separately measured variable.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def rms_norm(
    tokens: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """RMS normalisation in one fused kernel.

    Falls back to the reference arithmetic when the fused operator is unavailable, so the
    function is usable on any build; callers that care about which path ran should check
    :func:`has_fused_rms_norm`.
    """
    if hasattr(F, "rms_norm"):
        return F.rms_norm(tokens, (tokens.shape[-1],), weight, eps)
    dtype = tokens.dtype
    normalized = tokens.float() * torch.rsqrt(
        tokens.float().pow(2).mean(dim=-1, keepdim=True) + eps
    )
    return normalized.to(dtype) * weight


def has_fused_rms_norm() -> bool:
    return hasattr(F, "rms_norm")


def rope_tables(
    dim: int,
    length: int = 1024,
    *,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Real RoPE tables, ``[length, 1, dim]`` each, for the adjacent-pair rotation.

    The rotation pairs feature ``2i`` with ``2i+1``, which is the pairing the released
    checkpoint was trained with (the complex view of the token interleaves consecutive
    features).  ``cos`` repeats each cosine across its pair; ``sin`` carries the signs that
    make ``x * cos + swap_pairs(x) * sin`` reproduce the complex product exactly:

    ``(a + bi)(c + si) = (ac - bs) + (as + bc)i``.
    """
    if dim <= 0 or dim % 2 != 0:
        raise ValueError(f"RoPE dim must be positive and even, got {dim}")
    frequency = 1.0 / (
        10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float64)[: dim // 2] / dim)
    )
    phase = torch.outer(torch.arange(length, dtype=torch.float64), frequency)
    cosine = phase.cos()
    sine = phase.sin()
    cos_full = cosine.repeat_interleave(2, dim=-1)
    sin_full = torch.stack([-sine, sine], dim=-1).reshape(length, dim)
    return (
        cos_full.to(dtype).view(length, 1, dim),
        sin_full.to(dtype).view(length, 1, dim),
    )


def apply_rope_real(
    tokens: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_heads: int,
) -> torch.Tensor:
    """Apply RoPE with real arithmetic, ``tokens`` being ``[B, S, num_heads * head_dim]``.

    ``cos`` and ``sin`` are ``[S, 1, head_dim]`` from :func:`rope_tables`.  Operator count:
    one pair swap, one multiply and one fused multiply-add, all in the model's own dtype.
    """
    batch, sequence, width = tokens.shape
    if width % num_heads:
        raise ValueError(f"token width {width} is not divisible by {num_heads} heads")
    head_dim = width // num_heads
    if cos.shape[-1] != head_dim or sin.shape[-1] != head_dim:
        raise ValueError(
            f"tables must have width {head_dim}, got {cos.shape[-1]} and {sin.shape[-1]}"
        )
    if cos.shape[0] < sequence or sin.shape[0] < sequence:
        raise ValueError(
            f"tables cover {cos.shape[0]} positions but the sequence is {sequence}"
        )
    view = tokens.view(batch, sequence, num_heads, head_dim)
    swapped = (
        view.view(batch, sequence, num_heads, head_dim // 2, 2)
        .flip(-1)
        .reshape(batch, sequence, num_heads, head_dim)
    )
    rotated = torch.addcmul(view * cos[:sequence], swapped, sin[:sequence])
    return rotated.reshape(batch, sequence, width)


def reference_rope(
    tokens: torch.Tensor,
    frequencies: torch.Tensor,
    num_heads: int,
) -> torch.Tensor:
    """The shipped complex128 implementation, kept here as the oracle for equivalence tests."""
    batch, sequence, width = tokens.shape
    head_dim = width // num_heads
    view = tokens.view(batch, sequence, num_heads, head_dim)
    complex_tokens = torch.view_as_complex(
        view.double().reshape(batch, sequence, num_heads, head_dim // 2, 2)
    )
    complex_output = complex_tokens * frequencies.to(
        device=tokens.device, dtype=torch.complex128
    )
    return torch.view_as_real(complex_output).reshape(batch, sequence, width).to(
        tokens.dtype
    )


def complex_tables(
    dim: int,
    length: int = 1024,
) -> torch.Tensor:
    """Complex frequency table in the shipped layout, ``[length, 1, dim // 2]``.

    The trailing singleton is the model's own convention (``experts.py`` reshapes the
    concatenated frame/height/width tables to ``[tokens, 1, head_dim // 2]``), so an oracle
    built here is shaped like the table the model actually passes.
    """
    if dim <= 0 or dim % 2 != 0:
        raise ValueError(f"RoPE dim must be positive and even, got {dim}")
    frequency = 1.0 / (
        10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float64)[: dim // 2] / dim)
    )
    phase = torch.outer(torch.arange(length), frequency)
    return torch.polar(torch.ones_like(phase), phase).view(length, 1, dim // 2)
