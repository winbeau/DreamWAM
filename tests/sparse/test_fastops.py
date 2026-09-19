"""Numerical equivalence of the fused operators against the shipped implementations.

The replacements exist to remove casts and memory traffic, not to change the model, so the
test is equality against the original rather than a tolerance chosen for convenience.
"""

from __future__ import annotations

import pytest
import torch

from dreamwam.sparse.fastops import (
    apply_rope_real,
    complex_tables,
    has_fused_rms_norm,
    reference_rope,
    rms_norm,
    rope_tables,
)

HEADS = 4
HEAD_DIM = 8
WIDTH = HEADS * HEAD_DIM
SEQUENCE = 6


def reference_rms_norm(tokens, weight, eps):
    dtype = tokens.dtype
    normalized = tokens.float() * torch.rsqrt(
        tokens.float().pow(2).mean(dim=-1, keepdim=True) + eps
    )
    return normalized.to(dtype) * weight


# --- RMSNorm ----------------------------------------------------------------------


def test_rms_norm_matches_the_shipped_arithmetic():
    torch.manual_seed(0)
    tokens = torch.randn(2, 5, 16)
    weight = torch.randn(16)
    got = rms_norm(tokens, weight, 1e-5)
    expected = reference_rms_norm(tokens, weight, 1e-5)
    assert torch.allclose(got, expected, atol=1e-6), (got - expected).abs().max()


def test_rms_norm_is_scale_invariant_like_the_original():
    """RMS normalisation must not care about the input's overall scale."""
    tokens = torch.randn(1, 3, 8)
    weight = torch.ones(8)
    base = rms_norm(tokens, weight, 1e-5)
    scaled = rms_norm(tokens * 100.0, weight, 1e-5)
    assert torch.allclose(base, scaled, atol=1e-5)


def test_rms_norm_reports_which_path_is_available():
    """The fused operator is a torch build property, so the caller must be able to ask."""
    assert isinstance(has_fused_rms_norm(), bool)


def test_rms_norm_keeps_the_input_dtype():
    tokens = torch.randn(1, 3, 8, dtype=torch.bfloat16)
    weight = torch.ones(8, dtype=torch.bfloat16)
    assert rms_norm(tokens, weight, 1e-5).dtype == torch.bfloat16


# --- RoPE -------------------------------------------------------------------------


def test_real_rope_matches_the_complex_reference():
    torch.manual_seed(0)
    tokens = torch.randn(1, SEQUENCE, WIDTH, dtype=torch.float64).float()
    cos, sin = rope_tables(HEAD_DIM, length=SEQUENCE, dtype=torch.float32)
    got = apply_rope_real(tokens, cos, sin, HEADS)
    expected = reference_rope(
        tokens, complex_tables(HEAD_DIM, length=SEQUENCE), HEADS
    )
    assert got.shape == tokens.shape
    assert torch.allclose(got, expected, atol=1e-6), (got - expected).abs().max()


def test_real_rope_preserves_the_pairing_the_checkpoint_was_trained_with():
    """A pure rotation must not change vector magnitudes within each pair."""
    torch.manual_seed(1)
    tokens = torch.randn(1, SEQUENCE, WIDTH)
    cos, sin = rope_tables(HEAD_DIM, length=SEQUENCE)
    rotated = apply_rope_real(tokens, cos, sin, HEADS)
    before = tokens.view(SEQUENCE, HEADS, HEAD_DIM // 2, 2).pow(2).sum(-1)
    after = rotated.view(SEQUENCE, HEADS, HEAD_DIM // 2, 2).pow(2).sum(-1)
    assert torch.allclose(before, after, atol=1e-5)


def test_the_half_split_convention_would_give_a_different_answer():
    """Guards against silently implementing the common `rotate_half` variant, which pairs
    feature i with feature i + head_dim/2 while the trained checkpoint pairs 2i with 2i+1."""
    torch.manual_seed(2)
    tokens = torch.randn(1, SEQUENCE, WIDTH)
    cos, sin = rope_tables(HEAD_DIM, length=SEQUENCE)
    correct = apply_rope_real(tokens, cos, sin, HEADS)
    head = tokens.view(1, SEQUENCE, HEADS, HEAD_DIM)
    half = HEAD_DIM // 2
    rotate_half = torch.cat([-head[..., half:], head[..., :half]], dim=-1)
    wrong = (head * cos + rotate_half * sin).reshape(1, SEQUENCE, WIDTH)
    assert not torch.allclose(correct, wrong, atol=1e-4), (
        "the two RoPE conventions must not coincide, or this test proves nothing"
    )


def test_rope_tables_have_the_documented_shape_and_layout():
    cos, sin = rope_tables(HEAD_DIM, length=SEQUENCE)
    assert cos.shape == (SEQUENCE, 1, HEAD_DIM)
    assert sin.shape == (SEQUENCE, 1, HEAD_DIM)
    # cos repeats across each pair; sin alternates sign within it.
    assert torch.allclose(cos[..., 0::2], cos[..., 1::2])
    assert torch.allclose(sin[..., 0::2], -sin[..., 1::2])


def test_rope_at_position_zero_is_the_identity():
    cos, sin = rope_tables(HEAD_DIM, length=SEQUENCE)
    tokens = torch.randn(1, 1, WIDTH)
    rotated = apply_rope_real(tokens, cos, sin, HEADS)
    assert torch.allclose(rotated, tokens, atol=1e-6), "phase 0 must not rotate anything"


def test_rope_validates_its_inputs():
    cos, sin = rope_tables(HEAD_DIM, length=SEQUENCE)
    with pytest.raises(ValueError, match="divisible"):
        apply_rope_real(torch.randn(1, SEQUENCE, WIDTH + 1), cos, sin, HEADS)
    with pytest.raises(ValueError, match="width"):
        apply_rope_real(torch.randn(1, SEQUENCE, WIDTH), cos, sin * 0 + 1, HEADS - 1)
    with pytest.raises(ValueError, match="positions"):
        apply_rope_real(torch.randn(1, SEQUENCE + 4, WIDTH), cos, sin, HEADS)
    with pytest.raises(ValueError, match="even"):
        rope_tables(7, length=SEQUENCE)
