#!/usr/bin/env python3
"""Does action-guided visual token/FFN compression actually save request time?

S1 showed the bottleneck is not attention but the video tokens' projections and FFN: about
196 ms of the 514 ms request is ``linear``/``addmm``, and the FFN alone is ~61% of the video
expert's FLOPs. VV sparsity cannot touch that (S3), so the question becomes whether computing
the block for *fewer video tokens* converts into wall time.

Three mechanisms, all at the released geometry (30 layers, 10 steps, hidden 3072, ffn 14336,
24 heads x 128):

``block_full(n)``     the whole video block for ``n`` tokens - token dropping upper bound
``ffn_only(n)``       just the FFN, to separate it from the rest of the block
``block_sparse_ffn``  attention and projections on all 294 tokens, FFN only on ``k`` of them
                      (shape-preserving, so unpatchify and residual structure are untouched)

Every variant keeps the real gather/scatter cost of selecting token rows. Reported saving is
against the dense 30x10 layer-steps and expressed as a fraction of the measured 514 ms
complete request, so a result is directly comparable with S1 and S3.

    CUDA_VISIBLE_DEVICES=7 .venv/bin/python scripts/sparse/benchmark_token_compression.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from dreamwam.layers import DiTBlock, apply_rope, modulate

HIDDEN = 3072
FFN = 14336
HEADS = 24
HEAD_DIM = 128
TOKENS_PER_FRAME = 98
NUM_FRAMES = 3
VIDEO_TOKENS = TOKENS_PER_FRAME * NUM_FRAMES
ACTION_TOKENS = 32
CONTEXT_TOKENS = 129
LAYERS = 30
STEPS = 10


def make_blocks(layers: int, device, dtype):
    blocks = []
    for _ in range(layers):
        block = DiTBlock(
            hidden_dim=HIDDEN,
            attn_head_dim=HEAD_DIM,
            num_heads=HEADS,
            ffn_dim=FFN,
            eps=1e-6,
        ).to(device=device, dtype=dtype)
        # Constant fill keeps allocation fast while staying finite; GEMM time on tensor
        # cores is data independent, and every variant sees the same weights.
        for parameter in block.parameters():
            parameter.data.fill_(0.02)
        block.eval()
        blocks.append(block)
    return blocks


class BlockRunner:
    """The video expert's per-layer computation, mirroring mot.py's two helpers."""

    def __init__(self, blocks, device, dtype):
        self.blocks = blocks
        self.device = device
        self.dtype = dtype
        self.freqs = torch.polar(
            torch.ones(VIDEO_TOKENS, HEAD_DIM // 2, dtype=torch.float64, device=device),
            torch.zeros(VIDEO_TOKENS, HEAD_DIM // 2, dtype=torch.float64, device=device),
        ).view(VIDEO_TOKENS, 1, -1)
        self.modulation = torch.full(
            (1, VIDEO_TOKENS, 6, HIDDEN), 0.01, device=device, dtype=dtype
        )
        self.context = torch.full(
            (1, CONTEXT_TOKENS, HIDDEN), 0.02, device=device, dtype=dtype
        )
        self.context_mask = torch.ones(
            1, VIDEO_TOKENS, CONTEXT_TOKENS, dtype=torch.bool, device=device
        )
        self.mask = torch.ones(
            VIDEO_TOKENS, VIDEO_TOKENS, dtype=torch.bool, device=device
        )
        self.mask[:TOKENS_PER_FRAME, TOKENS_PER_FRAME:] = False
        # Precomputed so the FFN-only variant does not allocate per layer.
        self.constant_modulation = torch.full(
            (1, VIDEO_TOKENS, HIDDEN), 0.01, device=device, dtype=dtype
        )

    def _attention(self, block, tokens):
        values = (block.modulation + self.modulation[:1, : tokens.shape[1]]).chunk(6, dim=2)
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = (
            value.squeeze(2) for value in values
        )
        attention_input = modulate(block.norm1(tokens), shift_a, scale_a)
        query = apply_rope(
            block.self_attn.norm_q(block.self_attn.q(attention_input)),
            self.freqs[: tokens.shape[1]],
            HEADS,
        )
        key = apply_rope(
            block.self_attn.norm_k(block.self_attn.k(attention_input)),
            self.freqs[: tokens.shape[1]],
            HEADS,
        )
        value = block.self_attn.v(attention_input)
        length = tokens.shape[1]
        mixed = F.scaled_dot_product_attention(
            query.view(1, length, HEADS, HEAD_DIM).transpose(1, 2),
            key.view(1, length, HEADS, HEAD_DIM).transpose(1, 2),
            value.view(1, length, HEADS, HEAD_DIM).transpose(1, 2),
            attn_mask=self.mask[:length, :length],
        ).transpose(1, 2).reshape(1, length, HIDDEN)
        tokens = tokens + gate_a * block.self_attn.o(mixed)
        tokens = tokens + block.cross_attn(
            block.norm3(tokens), self.context, self.context_mask[:, :length]
        )
        return tokens, shift_f, scale_f, gate_f

    def block_full(self, tokens):
        """Whole block for every token (token-dropping upper bound)."""
        for block in self.blocks:
            tokens, shift_f, scale_f, gate_f = self._attention(block, tokens)
            ffn_input = modulate(block.norm2(tokens), shift_f, scale_f)
            tokens = tokens + gate_f * block.ffn(ffn_input)
        return tokens

    def ffn_only(self, tokens):
        for block in self.blocks:
            length = tokens.shape[1]
            ffn_input = modulate(
                block.norm2(tokens),
                self.constant_modulation[:, :length],
                self.constant_modulation[:, :length],
            )
            tokens = tokens + block.ffn(ffn_input)
        return tokens

    def block_sparse_ffn(self, tokens, keep: int):
        """Attention and projections dense; FFN computed on ``keep`` rows only."""
        for block in self.blocks:
            tokens, shift_f, scale_f, gate_f = self._attention(block, tokens)
            index = torch.arange(keep, device=tokens.device)
            selected = tokens.index_select(1, index)
            selected_input = modulate(
                block.norm2(selected),
                shift_f.index_select(1, index),
                scale_f.index_select(1, index),
            )
            update = gate_f.index_select(1, index) * block.ffn(selected_input)
            tokens = tokens.index_copy(1, index, selected + update)
        return tokens


def time_call(function, *, warmup: int, iters: int, device) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        function()
    end.record()
    torch.cuda.synchronize(device)
    return start.elapsed_time(end) / iters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--request-ms", type=float, default=514.0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    blocks = make_blocks(LAYERS, device, dtype)
    runner = BlockRunner(blocks, device, dtype)
    dense_tokens = torch.full((1, VIDEO_TOKENS, HIDDEN), 0.02, device=device, dtype=dtype)

    report: dict = {
        "script": "scripts/sparse/benchmark_token_compression.py",
        "stage": "S3b-token-compression-ceiling",
        "status": "MEASURED",
        "device": torch.cuda.get_device_name(0),
        "dtype": args.dtype,
        "geometry": {
            "video_tokens": VIDEO_TOKENS,
            "tokens_per_frame": TOKENS_PER_FRAME,
            "layers": LAYERS,
            "steps": STEPS,
            "layer_steps": LAYERS * STEPS,
            "hidden": HIDDEN,
            "ffn": FFN,
        },
        "request_ms_reference": args.request_ms,
        "request_fraction_per_layer_step": 1.0 / (LAYERS * STEPS),
        "variants": [],
    }

    def record(label, ms, note=None):
        # One call runs all 30 layers, i.e. exactly one denoising step.
        per_request = ms * STEPS
        entry = {
            "variant": label,
            "ms_per_step_30_layers": ms,
            "ms_per_request": per_request,
            "note": note,
        }
        report["variants"].append(entry)
        return per_request

    dense_step = time_call(
        lambda: runner.block_full(dense_tokens),
        warmup=args.warmup,
        iters=args.iters,
        device=device,
    )
    dense_request = record("block_full_294", dense_step, "dense reference for one step")

    for keep in (294, 220, 176, 147, 98, 49):
        tokens = torch.full((1, keep, HIDDEN), 0.02, device=device, dtype=dtype)
        ms = time_call(
            lambda tokens=tokens: runner.block_full(tokens),
            warmup=args.warmup,
            iters=args.iters,
            device=device,
        )
        record(f"block_full_{keep}", ms, "whole block on fewer tokens (needs shape handling)")

    ffn_dense = time_call(
        lambda: runner.ffn_only(dense_tokens),
        warmup=args.warmup,
        iters=args.iters,
        device=device,
    )
    record("ffn_only_294", ffn_dense, "FFN cost alone")

    for keep in (220, 176, 147, 98, 49):
        ms = time_call(
            lambda keep=keep: runner.block_sparse_ffn(dense_tokens, keep),
            warmup=args.warmup,
            iters=args.iters,
            device=device,
        )
        record(
            f"block_sparse_ffn_keep{keep}",
            ms,
            "attention/projections dense, FFN on keep rows (shape preserving)",
        )

    for entry in report["variants"]:
        saving = dense_request - entry["ms_per_request"]
        entry["saving_ms_per_request"] = saving
        entry["saving_fraction_of_request"] = saving / args.request_ms
        entry["speedup_bound"] = args.request_ms / (args.request_ms - saving)

    # Representativeness check: S1 measured a 48.8 ms denoising step and ~196 ms of
    # linear/addmm across the request. If block_full_294 is far outside that range this
    # reimplementation is not a faithful stand-in and its numbers must not be quoted.
    report["representativeness"] = {
        "measured_step_ms_s1": 48.84,
        "measured_gemm_ms_per_request_s1": 195.5,
        "block_full_294_ms_per_step": dense_step,
        "ratio_to_measured_step": dense_step / 48.84,
    }

    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
