#!/usr/bin/env python3
"""S2 acceptance: operator parity for the Sparse-WAM reference on the real model.

Loads the released checkpoint once and re-runs the *same* request through four
configurations that must agree where the mathematics says they must:

``dense``        the shipped fused joint attention (no sparse code on the path)
``split``        video rows and action rows separated, every legal key kept
``full_masked``  routed path, ``selection=all``: every legal VV key, masked backend
``full_gather``  as above with the gather backend (exact same key set, different kernel)

The first three must agree to numerical tolerance; ``full_gather`` must agree with them as
well, otherwise the gather/index path is wrong.  Only after this passes does a restricted
``future_ratio`` number mean anything, so the restricted variants are reported here as
diagnostics with an explicit tolerance rather than as a pass/fail gate.

    CUDA_VISIBLE_DEVICES=7 .venv/bin/python scripts/sparse/check_parity.py --out parity.json
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


def compare(reference: np.ndarray, candidate: np.ndarray) -> dict:
    difference = np.abs(reference - candidate)
    denominator = np.linalg.norm(reference)
    return {
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
        "relative_l2": float(np.linalg.norm(reference - candidate) / max(denominator, 1e-12)),
        "gripper_sign_flips": int(
            (np.sign(reference[:, -1]) != np.sign(candidate[:, -1])).sum()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dreamwam_joint.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=5e-3,
        help="max-abs tolerance for the full-budget controls (bf16 kernels differ)",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    from dreamwam.config import load_release_config
    from dreamwam.policy import build_policy

    config = load_release_config(args.config)
    policy = build_policy(config, device=args.device)
    images, state, instruction = build_inputs(args.seed, int(policy.image_size))

    def run(sparse: dict | None) -> np.ndarray:
        policy.sparse_config = SparseConfig.from_mapping(sparse)
        return policy.predict_action(images=images, state=state, instruction=instruction)

    runs: dict[str, np.ndarray] = {}
    runs["dense"] = run(None)
    runs["split"] = run(
        {"enabled": True, "selection": "all", "backend": "gather", "future_ratio": 1.0}
    )
    runs["full_masked"] = run(
        {"enabled": True, "selection": "all", "backend": "masked", "future_ratio": 1.0}
    )
    runs["full_gather"] = run(
        {"enabled": True, "selection": "all", "backend": "gather", "future_ratio": 1.0}
    )

    diagnostics: dict[str, np.ndarray] = {}
    for label, payload in {
        "av_ratio_0.50": {"selection": "av", "future_ratio": 0.50},
        "av_context_ratio_0.50": {"selection": "av_context", "future_ratio": 0.50},
        "av_ratio_0.25": {"selection": "av", "future_ratio": 0.25},
        "recency_ratio_0.50": {"selection": "recency", "future_ratio": 0.50},
        "av_ratio_0.00": {"selection": "av", "future_ratio": 0.00},
    }.items():
        diagnostics[label] = run(
            {"enabled": True, "backend": "masked", **payload}
        )

    report: dict = {
        "script": "scripts/sparse/check_parity.py",
        "stage": "S2-operator-parity",
        "status": "MEASURED",
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "config": str(args.config),
        "checkpoint": str(config.paths.checkpoint),
        "tolerance_max_abs": args.tolerance,
        "reference": "dense",
        "controls": {},
        "restricted_diagnostics": {},
        "density": {},
    }
    for label in ("split", "full_masked", "full_gather"):
        report["controls"][label] = compare(runs["dense"], runs[label])
    for label, actions in diagnostics.items():
        report["restricted_diagnostics"][label] = compare(runs["dense"], actions)

    # Density actually executed by each restricted variant, from the model's own counters.
    for label, payload in {
        "av_ratio_0.50": {"selection": "av", "future_ratio": 0.50},
        "av_ratio_0.25": {"selection": "av", "future_ratio": 0.25},
        "av_ratio_0.00": {"selection": "av", "future_ratio": 0.00},
        "recency_ratio_0.50": {"selection": "recency", "future_ratio": 0.50},
    }.items():
        policy.sparse_config = SparseConfig.from_mapping(
            {"enabled": True, "backend": "masked", **payload}
        )
        policy.predict_action(images=images, state=state, instruction=instruction)
        calls = policy.model.mot.sparse_diagnostics.get("calls", 0.0)
        if calls:
            report["density"][label] = {
                "mean_density": policy.model.mot.sparse_diagnostics["density_sum"] / calls,
                "mean_fallback_fraction": policy.model.mot.sparse_diagnostics["fallback_sum"]
                / calls,
                "calls": calls,
                "last": policy.model.mot.sparse_diagnostics.get("last"),
            }

    failures = [
        label
        for label, metrics in report["controls"].items()
        if metrics["max_abs"] > args.tolerance
    ]
    report["controls_pass"] = not failures
    report["control_failures"] = failures

    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")
    if failures:
        raise SystemExit(f"parity controls failed: {failures}")


if __name__ == "__main__":
    main()
