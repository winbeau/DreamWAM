#!/usr/bin/env python3
"""Measure how much of the request is per-operator dispatch overhead.

The hypothesis this tests comes from three measurements that a compute-bound model cannot
produce: the video block costs the same for 1 token as for 294, every sparse variant is
slower than the dense call it replaces, and effective throughput is ~10% of the device's
capability. All three follow if the run is dominated by per-operator cost - eager dispatch
and kernel launch - rather than by arithmetic: removing FLOPs then changes nothing, and
adding operators makes things worse.

If that is right, the lever is to stop dispatching tens of thousands of small operators, so
this script measures the headroom of that directly against the *unmodified* model:

``baseline``      the shipped eager path through ``sample_action``
``compile_*``     torch.compile on the same call, default and reduce-overhead (CUDA graphs)
``cuda_graph``    a hand-captured graph of the same call replayed from static buffers

Correctness is checked at every step: a faster path that changes the action is not a result.
Failures are reported rather than hidden, because a failed capture is itself information
about where the cost lives.

    CUDA_VISIBLE_DEVICES=7 .venv/bin/python scripts/sparse/benchmark_dispatch_overhead.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from dreamwam.preprocessing.libero import PROMPT_TEMPLATE


def build_inputs(seed: int, image_size: int):
    rng = np.random.default_rng(seed)
    images = {
        "agentview": rng.integers(0, 255, (image_size, image_size, 3), dtype=np.uint8),
        "wrist": rng.integers(0, 255, (image_size, image_size, 3), dtype=np.uint8),
    }
    state = np.zeros(8, dtype=np.float32)
    return images, state, "pick up the black bowl and place it on the plate"


def timed(function, *, warmup: int, reps: int) -> dict:
    """Wall-clock request timings with the device synchronised outside the measurement."""
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        torch.cuda.synchronize()
        start = time.perf_counter()
        function()
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - start)
    samples.sort()
    return {
        "min": samples[0],
        "p50": samples[len(samples) // 2],
        "mean": statistics.fmean(samples),
        "max": samples[-1],
        "reps": len(samples),
    }


def compare(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    difference = (reference - candidate).abs()
    return {
        "max_abs": float(difference.max()),
        "relative_l2": float(
            difference.norm() / reference.norm().clamp(min=1e-12)
        ),
        "identical": bool(torch.equal(reference, candidate)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dreamwam_joint.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--reps", type=int, default=8)
    parser.add_argument("--compile-warmup", type=int, default=3)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    from dreamwam.config import load_release_config
    from dreamwam.policy import build_policy

    config = load_release_config(args.config)
    policy = build_policy(config, device=args.device)
    model = policy.model
    evaluation = policy.evaluation
    images, state, instruction = build_inputs(args.seed, int(policy.image_size))

    # Replicate predict_action's preparation once; every variant below starts from it, so the
    # only difference between variants is how the denoising loop is executed.
    proprio = policy.normalizer.normalize_state(
        torch.from_numpy(state).unsqueeze(0)
    ).to(device=policy.device, dtype=policy.dtype)
    context, context_mask = policy.text_encoder([PROMPT_TEMPLATE.format(task=instruction)])
    first_frame = policy._encode_first_frame(images)
    temporal = int(config.model.vae_temporal_downsample_factor)
    video_frames = int(evaluation["video_frames"])
    latent_frames = (video_frames - 1) // temporal + 1

    def sample(**overrides):
        kwargs = dict(
            first_frame_latents=first_frame,
            context=context,
            context_mask=context_mask,
            proprio=proprio,
            action_horizon=int(evaluation["action_horizon"]),
            num_video_latent_frames=latent_frames,
            num_steps=int(evaluation["denoising_steps"]),
            seed=int(evaluation["seed"]),
            rand_device=str(evaluation["rand_device"]),
        )
        kwargs.update(overrides)
        return model.sample_action(**kwargs)

    report: dict = {
        "script": "scripts/sparse/benchmark_dispatch_overhead.py",
        "stage": "route3-dispatch-overhead",
        "status": "MEASURED",
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "config": str(args.config),
        "checkpoint": str(config.paths.checkpoint),
        "denoising_steps": int(evaluation["denoising_steps"]),
        "hypothesis": (
            "per-operator dispatch dominates, so removing operators (not FLOPs) is the lever"
        ),
        "variants": {},
    }

    reference = sample()
    if not torch.isfinite(reference).all():
        raise FloatingPointError("baseline produced non-finite actions")
    report["variants"]["baseline"] = {
        "timing": timed(lambda: sample(), warmup=args.warmup, reps=args.reps)
    }

    for mode in ("default", "reduce-overhead"):
        label = f"compile_{mode.replace('-', '_')}"
        try:
            compiled = torch.compile(sample, mode=mode, dynamic=False)
            produce = lambda compiled=compiled: compiled()  # noqa: E731
            for _ in range(args.compile_warmup):
                candidate = produce()
            torch.cuda.synchronize()
            report["variants"][label] = {
                "timing": timed(produce, warmup=args.warmup, reps=args.reps),
                "correctness": compare(reference, candidate),
            }
        except Exception as error:  # noqa: BLE001 - a failed variant is a result
            report["variants"][label] = {
                "error": f"{type(error).__name__}: {error}",
            }

    try:
        static_first = first_frame.clone()
        static_context = context.clone()
        static_mask = context_mask.clone()
        static_proprio = proprio.clone()

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                sample(
                    first_frame_latents=static_first,
                    context=static_context,
                    context_mask=static_mask,
                    proprio=static_proprio,
                )
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_output = sample(
                first_frame_latents=static_first,
                context=static_context,
                context_mask=static_mask,
                proprio=static_proprio,
            )
        graph.replay()
        torch.cuda.synchronize()
        captured = static_output.clone()
        report["variants"]["cuda_graph"] = {
            "timing": timed(graph.replay, warmup=args.warmup, reps=args.reps),
            "correctness": compare(reference, captured),
            "note": (
                "replay reuses the captured initial noise, which matches the shipped path "
                "because sample_action re-seeds from its own config on every call"
            ),
        }
    except Exception as error:  # noqa: BLE001
        report["variants"]["cuda_graph"] = {"error": f"{type(error).__name__}: {error}"}

    baseline = report["variants"]["baseline"]["timing"]["p50"]
    report["speedups"] = {
        label: (
            baseline / entry["timing"]["p50"]
            if "timing" in entry and entry["timing"]["p50"] > 0
            else None
        )
        for label, entry in report["variants"].items()
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
