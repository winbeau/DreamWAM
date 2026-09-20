#!/usr/bin/env python3
"""Generate a finite hybrid search manifest without torch, weights or GPU access."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dreamwam.sparse.hybrid import HybridConfig
from dreamwam.sparse.hybrid.search import generate_sweep, parse_indices


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-steps", type=int, default=10)
    p.add_argument("--dense-steps", default="0")
    p.add_argument("--candidate-sparse-steps", default="1:10")
    p.add_argument("--refresh-counts", default="0,1,2")
    recompute = p.add_mutually_exclusive_group()
    recompute.add_argument("--recompute-ratio", type=float, default=0.1)
    recompute.add_argument("--recompute-ratios", help="comma-separated finite budget grid")
    p.add_argument("--read-mode", choices=("full", "compact"), default="compact")
    read = p.add_mutually_exclusive_group()
    read.add_argument("--read-ratio", type=float, default=0.25)
    read.add_argument("--read-ratios", help="comma-separated finite budget grid")
    selectors = p.add_mutually_exclusive_group()
    selectors.add_argument("--selection", choices=("uniform", "drift", "action_drift", "action", "action_context", "visual_context"), default="action_drift")
    selectors.add_argument("--selections", help="comma-separated selector grid")
    p.add_argument("--guidance-weight", type=float, default=1.0)
    p.add_argument("--context-weight", type=float, default=1.0)
    p.add_argument("--support-seed-ratio", type=float, default=0.1)
    p.add_argument("--reuse-mode", choices=("features", "structure"), default="features")
    p.add_argument("--frame-quota", choices=("none", "balanced"),
                   help="full defaults to global queries; use balanced for matched compact-read ablations")
    p.add_argument("--backend", choices=("eager", "buffered", "cuda_graph"), default="eager")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-candidates", type=int, default=512,
                   help="reject, never truncate, a larger Cartesian search")
    args = p.parse_args()
    try:
        selection = dict(method=args.selections.split(",")[0] if args.selections is not None else args.selection,
                         guidance_weight=args.guidance_weight,
                         context_weight=args.context_weight, support_seed_ratio=args.support_seed_ratio)
        if args.frame_quota is not None:
            selection["frame_quota"] = args.frame_quota
        config = HybridConfig.from_mapping(dict(
            schedule=dict(kind="periodic", num_steps=args.num_steps, refresh_every=args.num_steps),
            recompute=dict(keep_ratio=float(args.recompute_ratios.split(",")[0])
                          if args.recompute_ratios is not None else args.recompute_ratio),
            read=dict(mode=args.read_mode, keep_ratio=float(args.read_ratios.split(",")[0])
                      if args.read_ratios is not None else args.read_ratio),
            selection=selection, execution=dict(backend=args.backend), reuse=dict(mode=args.reuse_mode)))
        candidates = generate_sweep(config, parse_indices(args.dense_steps),
            parse_indices(args.candidate_sparse_steps), parse_indices(args.refresh_counts),
            read_ratios=None if args.read_ratios is None else tuple(map(float, args.read_ratios.split(","))),
            recompute_ratios=None if args.recompute_ratios is None else tuple(map(float, args.recompute_ratios.split(","))),
            selections=None if args.selections is None else tuple(args.selections.split(",")),
            max_candidates=args.max_candidates)
    except (ValueError, TypeError) as exc:
        p.error(str(exc))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        for row in candidates:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(dict(candidates=len(candidates), output=str(args.out), gpu_used=False)))


if __name__ == "__main__":
    main()
