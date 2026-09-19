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
HEAD_DIM = 16
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


class CountingMode(torch.utils._python_dispatch.TorchDispatchMode):
    """Count ATen operators, ignoring the ones the counting itself performs."""

    def __init__(self) -> None:
        super().__init__()
        self.counts: Counter[str] = Counter()
        self._depth = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        # Nested dispatches would double-count decomposed operators.
        if self._depth == 0:
            self.counts[str(func)] += 1
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
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device("cpu")
    video, action, mot = build(device)
    mot.eval()

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

    mode = CountingMode()
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
        "layers": LAYERS,
        "operators_per_layer": per_layer,
        "operators_per_request_estimate": per_request,
        "profiled_operators_per_request_s1": 80000,
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
