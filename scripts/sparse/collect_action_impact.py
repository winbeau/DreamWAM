#!/usr/bin/env python3
"""M1 calibration: measure the final-action impact of restricting one layer at a time.

This is the label source M1 is supposed to use. The paper's claim is that the visual
computation worth keeping is decided by its effect on the *action*, not on the generated
video, so an intervention has to be run to the final action rather than scored by a local
attention error.

Why one layer at a time: a single-layer restriction is the only attribution that can say
*which* layer carries decision-relevant visual interaction. Layers are restricted through
``sparse_layers``, which routes exactly that layer and leaves every other layer on the
shipped dense path, so the difference from the dense baseline is caused by that layer alone.

Cost is deliberately small: every configuration replays one fixed request through one
sampling call, so a full 30-layer profile costs 30 forwards - small enough to fit in a short
idle window on a shared machine. The companion per-head and per-stage sweeps reuse the same
harness.

    CUDA_VISIBLE_DEVICES=7 .venv/bin/python scripts/sparse/collect_action_impact.py \\
        --ratio 0.25 --out impact.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dreamwam.sparse import SparseConfig


def build_inputs(seed: int, image_size: int):
    rng = np.random.default_rng(seed)
    images = {
        "agentview": rng.integers(0, 255, (image_size, image_size, 3), dtype=np.uint8),
        "wrist": rng.integers(0, 255, (image_size, image_size, 3), dtype=np.uint8),
    }
    state = np.zeros(8, dtype=np.float32)
    return images, state, "pick up the black bowl and place it on the plate"


def components(reference: np.ndarray, candidate: np.ndarray) -> dict:
    difference = candidate - reference
    return {
        "max_abs": float(np.abs(difference).max()),
        "mean_abs": float(np.abs(difference).mean()),
        "relative_l2": float(
            np.linalg.norm(difference) / max(np.linalg.norm(reference), 1e-12)
        ),
        "translation_rms": float(np.sqrt((difference[:, :3] ** 2).mean())),
        "rotation_rms": float(np.sqrt((difference[:, 3:6] ** 2).mean())),
        "gripper_sign_flips": int(
            (np.sign(reference[:, -1]) != np.sign(candidate[:, -1])).sum()
        ),
        "gripper_fraction_unchanged": float(
            (np.sign(reference[:, -1]) == np.sign(candidate[:, -1])).mean()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dreamwam_joint.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.25,
        help="future-key ratio kept in the restricted layer (0.0 = conditioning frame only)",
    )
    parser.add_argument(
        "--selection", default="all", choices=("all", "av", "av_context", "recency", "uniform")
    )
    parser.add_argument("--backend", default="masked", choices=("masked", "gather"))
    parser.add_argument(
        "--layers",
        type=int,
        nargs="*",
        default=None,
        help="layers to sweep; default is every layer of the model",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    from dreamwam.config import load_release_config
    from dreamwam.policy import build_policy

    config = load_release_config(args.config)
    policy = build_policy(config, device=args.device)
    images, state, instruction = build_inputs(args.seed, int(policy.image_size))
    num_layers = int(config.model.num_layers)
    layers = list(range(num_layers)) if args.layers is None else list(args.layers)

    def run(sparse: dict | None) -> np.ndarray:
        policy.sparse_config = SparseConfig.from_mapping(sparse)
        return policy.predict_action(images=images, state=state, instruction=instruction)

    dense = run(None)
    if not np.isfinite(dense).all():
        raise FloatingPointError("dense reference produced non-finite actions")

    # A full-budget run through the routed path must reproduce dense; if it does not, the
    # interventions below are measuring the harness rather than the restriction.
    full = run(
        {
            "enabled": True,
            "selection": "all",
            "backend": args.backend,
            "future_ratio": 1.0,
            "sparse_layers": layers,
        }
    )
    control = components(dense, full)

    profile = []
    for layer in layers:
        action = run(
            {
                "enabled": True,
                "selection": args.selection,
                "backend": args.backend,
                "future_ratio": args.ratio,
                "block_size": 14,
                "sparse_layers": [layer],
            }
        )
        entry = {"layer": layer}
        entry.update(components(dense, action))
        policy.sparse_config = SparseConfig.from_mapping(
            {
                "enabled": True,
                "selection": args.selection,
                "backend": args.backend,
                "future_ratio": args.ratio,
                "block_size": 14,
                "sparse_layers": [layer],
            }
        )
        policy.predict_action(images=images, state=state, instruction=instruction)
        counters = policy.model.mot.sparse_diagnostics
        calls = counters.get("calls", 0.0)
        entry["executed_density"] = (
            counters.get("density_sum", 0.0) / calls if calls else None
        )
        entry["routed_calls"] = calls
        profile.append(entry)

    report = {
        "script": "scripts/sparse/collect_action_impact.py",
        "stage": "S4-action-impact",
        "status": "MEASURED",
        "device": torch.cuda.get_device_name(0),
        "checkpoint": str(config.paths.checkpoint),
        "setting": f"selection={args.selection} ratio={args.ratio} backend={args.backend}",
        "num_layers": num_layers,
        "full_budget_control": control,
        "control_pass": control["max_abs"] < 5e-3,
        "read_this_as": (
            "an intervention profile: which single layer's restricted VV key set moves the "
            "final action. It ranks layers for M1 and does not by itself certify any policy"
        ),
        "profile": profile,
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")
    if not report["control_pass"]:
        raise SystemExit("full-budget control failed; the profile is not interpretable")


if __name__ == "__main__":
    main()
