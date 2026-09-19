#!/usr/bin/env python3
"""Attribute operator counts to the code that emits them.

The dispatch hypothesis (docs/analysis/launch-bound-wam-sparsity.md 5b) says the request is
governed by how many operators run, not how much arithmetic they do. The profiler already
gives the totals for the released model - about 80k operator calls per request, dominated by
``aten::to``, ``copy_``, ``mul`` and ``add`` - but not which code emits them, and that is what
decides where a fix is worth writing.

Operator *counts* are almost independent of tensor size, so a small model on CPU produces the
same per-layer operator inventory as the released one at a fraction of the cost. This script
runs one joint forward under a counting dispatch mode and reports the breakdown by operator
and by the module that emitted it, then extrapolates to the released geometry.

    python scripts/sparse/count_operators.py            # CPU only, no checkpoint, no GPU
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

import torch

from dreamwam.experts import ActionDiT, VideoDiT
from dreamwam.mot import JointMoT
from dreamwam.sparse import SparseConfig

HIDDEN = 32
FFN = 64
HEADS = 2
# RoPE splits the head dimension into a frame part and two spatial parts, so the head
# dimension must be divisible by 3 with an even quotient (the released model uses 128).
HEAD_DIM = 24
LAYERS = 2
TOKENS_PER_FRAME = 4
FRAMES = 3
VIDEO_TOKENS = TOKENS_PER_FRAME * FRAMES
ACTION_TOKENS = 5
CONTEXT_TOKENS = 7
TEXT_DIM = 16
FREQ_DIM = 8
RELEASED_LAYERS = 30
RELEASED_STEPS = 10


def _originating_frame(depth: int = 24) -> str:
    """The innermost DreamWAM frame that asked for the operator.

    Metadata operators are cheap individually and only matter in aggregate, so the question is
    never "is this view expensive" but "which line emits sixty of them per layer". The
    innermost frame inside this repository is the answer.
    """
    import traceback

    for frame in reversed(traceback.extract_stack()[:-2][-depth:]):
        if "python_dispatch" in frame.filename:
            continue
        for marker in ("/dreamwam/", "/scripts/sparse/"):
            if marker in frame.filename:
                return f"{frame.filename.split(marker)[-1]}:{frame.lineno}"
    return "?"


class CountingMode(torch.utils._python_dispatch.TorchDispatchMode):
    """Count ATen operators, ignoring the ones the counting itself performs."""

    def __init__(self, trace: str | None = None) -> None:
        super().__init__()
        self.counts: Counter[str] = Counter()
        self.origins: Counter[str] = Counter()
        self.trace = trace
        self._depth = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        # Nested dispatches would double-count decomposed operators.
        name = str(func)
        if self._depth == 0:
            self.counts[name] += 1
            if self.trace and self.trace in name:
                self.origins[_originating_frame()] += 1
        self._depth += 1
        try:
            return func(*args, **(kwargs or {}))
        finally:
            self._depth -= 1


def build(device: torch.device):
    video = VideoDiT(
        video_latent_dim=4,
        flow_latent_dim=4,
        hidden_dim=HIDDEN,
        ffn_dim=FFN,
        text_dim=TEXT_DIM,
        freq_dim=FREQ_DIM,
        num_heads=HEADS,
        attn_head_dim=HEAD_DIM,
        num_layers=LAYERS,
        patch_size=(1, 2, 2),
        eps=1e-6,
    )
    action = ActionDiT(
        action_dim=4,
        hidden_dim=HIDDEN,
        ffn_dim=FFN,
        text_dim=TEXT_DIM,
        freq_dim=FREQ_DIM,
        num_heads=HEADS,
        attn_head_dim=HEAD_DIM,
        num_layers=LAYERS,
        eps=1e-6,
    )
    mot = JointMoT(
        video,
        action,
        action_attends_all_video=True,
        use_gradient_checkpointing=False,
    )
    return video, action, mot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sparse-json", default=None)
    parser.add_argument(
        "--trace",
        default=None,
        help="substring of an operator name to attribute to source lines, e.g. 'view'",
    )
    parser.add_argument(
        "--fast-ops",
        action="store_true",
        help="enable the fused RMSNorm and real-arithmetic RoPE before counting",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device("cpu")
    video, action, mot = build(device)
    mot.eval()

    switched_norms = 0
    if args.fast_ops:
        from dreamwam.layers import RMSNorm

        video.enable_fast_ops()
        action.enable_fast_ops()
        for module in mot.modules():
            if isinstance(module, RMSNorm):
                module.fused = True
                switched_norms += 1
        # A switch that matches nothing would make this comparison silently inert, which is
        # the failure mode this project has already hit twice. Fail loudly instead.
        if switched_norms == 0:
            raise RuntimeError(
                "fast ops requested but no RMSNorm module was switched; the comparison "
                "would be measuring the shipped path twice"
            )

    height, width = 2, 2  # patch grid per frame
    video_state = video.pre_dit(
        video_latents=torch.randn(1, 4, FRAMES, height * 2, width * 2),
        flow_latents=torch.zeros(1, 4, FRAMES, height * 2, width * 2),
        timestep=torch.zeros(1),
        context=torch.randn(1, CONTEXT_TOKENS, TEXT_DIM),
        context_mask=torch.ones(1, CONTEXT_TOKENS, dtype=torch.bool),
    )
    action_state = action.pre_dit(
        action_tokens=torch.randn(1, ACTION_TOKENS, 4),
        timestep=torch.zeros(1),
        context=torch.randn(1, CONTEXT_TOKENS, TEXT_DIM),
        context_mask=torch.ones(1, CONTEXT_TOKENS, dtype=torch.bool),
    )
    assert video_state["tokens"].shape[1] == VIDEO_TOKENS, video_state["tokens"].shape

    sparse = (
        SparseConfig.from_mapping(json.loads(args.sparse_json))
        if args.sparse_json
        else SparseConfig()
    )

    mode = CountingMode(args.trace)
    with torch.no_grad(), mode:
        mot(
            video_state=video_state,
            action_state=action_state,
            residual_injection=None,
            sparse=sparse,
            step_index=0,
            num_steps=RELEASED_STEPS,
        )

    counts = mode.counts
    total = sum(counts.values())
    per_layer = total / LAYERS
    per_request = per_layer * RELEASED_LAYERS * RELEASED_STEPS

    report = {
        "script": "scripts/sparse/count_operators.py",
        "stage": "route3-operator-attribution",
        "status": "MEASURED",
        "device": "cpu",
        "note": (
            "operator counts are essentially size-independent, so this small model inventories "
            "the same operators per layer as the released one"
        ),
        "sparse": sparse.describe(),
        "fast_ops": bool(args.fast_ops),
        "rms_norm_modules_switched": switched_norms,
        "layers": LAYERS,
        "operators_per_layer": per_layer,
        "operators_per_request_estimate": per_request,
        "profiled_operators_per_request_s1": 80000,
        "trace": args.trace,
        "traced_origins": [
            {"origin": origin, "per_layer": count / LAYERS, "share": count / total}
            for origin, count in mode.origins.most_common(20)
        ],
        "top_operators": [
            {"operator": name, "per_layer": count / LAYERS, "share": count / total}
            for name, count in counts.most_common(25)
        ],
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.out:
        from pathlib import Path

        Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
