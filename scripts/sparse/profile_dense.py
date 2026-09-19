#!/usr/bin/env python3
"""S1: Dense request decomposition for Sparse-WAM on DreamWAM Joint.

Answers one question before any sparse code is written: *how much of a complete
``predict_action`` request is self-attention, and how much of that is the VV part that
Sparse-WAM can actually sparsify?*  If that share is small, no amount of routing quality
turns VV sparsity into an end-to-end speedup, and the correct output of this script is a
negative result rather than an implementation.

Measured with CUDA events around the real call path (no re-implementation of
``predict_action``) plus an optional torch profiler kernel table:

* phase split        text encode / VAE encode / joint denoise / action post-processing
* attention split    self-attention (VV + AV + AA in one fused call) vs cross-attention
* kernel table       top CUDA kernels, to see whether the masked SDPA takes a slow path
* geometry           B, heads, head_dim, video/action token counts, layers, dtype

Nothing here changes model behaviour: the SDPA wrapper calls straight through and only
records events.  ``--no-instrument`` runs the same request without wrappers so the
instrumentation cost itself is measured rather than assumed.

Usage (on the evaluation server, from the DreamWAM checkout):

    .venv/bin/python scripts/sparse/profile_dense.py --config configs/dreamwam_joint.yaml
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from dreamwam.sparse import SparseConfig


class EventTimer:
    """Record CUDA-event elapsed time for wrapped calls, bucketed by kind."""

    def __init__(self) -> None:
        self.records: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {}
        self.calls: dict[str, int] = {}
        self.shapes: dict[str, tuple[int, ...]] = {}

    def wrap(self, function, kind: str):
        def wrapped(query, key, value, num_heads, attention_mask=None):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = function(query, key, value, num_heads, attention_mask)
            end.record()
            self.records.setdefault(kind, []).append((start, end))
            self.calls[kind] = self.calls.get(kind, 0) + 1
            self.shapes.setdefault(
                kind,
                (
                    int(query.shape[1]),
                    int(key.shape[1]),
                    int(query.shape[-1]),
                    int(num_heads),
                    -1 if attention_mask is None else int(attention_mask.ndim),
                ),
            )
            return output

        return wrapped

    def totals_ms(self) -> dict[str, float]:
        torch.cuda.synchronize()
        totals: dict[str, float] = {}
        for kind, pairs in self.records.items():
            totals[kind] = sum(
                start.elapsed_time(end) for start, end in pairs
            )
        return totals

    def reset(self) -> None:
        self.records.clear()
        self.calls.clear()
        self.shapes.clear()


class PhaseTimer:
    """Wall-clock phase timer around real policy methods."""

    def __init__(self) -> None:
        self.seconds: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def wrap(self, function, name: str):
        def wrapped(*args, **kwargs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = function(*args, **kwargs)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            self.seconds[name] = self.seconds.get(name, 0.0) + elapsed
            self.counts[name] = self.counts.get(name, 0) + 1
            return result

        return wrapped

    def reset(self) -> None:
        self.seconds.clear()
        self.counts.clear()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("no samples")
    index = min(int(round(fraction * (len(ordered) - 1))), len(ordered) - 1)
    return ordered[index]


def build_inputs(seed: int, image_size: int) -> tuple[dict, np.ndarray, str]:
    rng = np.random.default_rng(seed)
    images = {
        "agentview": rng.integers(0, 255, (image_size, image_size, 3), dtype=np.uint8),
        "wrist": rng.integers(0, 255, (image_size, image_size, 3), dtype=np.uint8),
    }
    state = np.zeros(8, dtype=np.float32)
    return images, state, "pick up the black bowl and place it on the plate"


def instrument(policy, timer: EventTimer, phases: PhaseTimer):
    """Wrap the real attention helper and the real policy phases."""
    import dreamwam.layers as layers_module
    import dreamwam.mot as mot_module

    original = layers_module.scaled_dot_product_attention

    def dispatch(query, key, value, num_heads, attention_mask=None):
        # Self-attention joins [video;action] against [video;action]; cross-attention
        # attends to the text/proprio context, so the key length differs.
        kind = "self_attention" if key.shape[1] == query.shape[1] else "cross_attention"
        return timer.wrap(original, kind)(query, key, value, num_heads, attention_mask)

    layers_module.scaled_dot_product_attention = dispatch
    mot_module.scaled_dot_product_attention = dispatch

    policy._encode_first_frame = phases.wrap(policy._encode_first_frame, "vae_encode")
    policy.text_encoder = phases.wrap(policy.text_encoder, "text_encode")
    policy.model.sample_action = phases.wrap(policy.model.sample_action, "joint_denoise")
    return original


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dreamwam_joint.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-instrument", action="store_true")
    parser.add_argument("--profile-kernels", action="store_true")
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--sparse-json",
        default=None,
        help="JSON sparse configuration applied to the loaded model",
    )
    parser.add_argument(
        "--denoising-steps",
        type=int,
        default=None,
        help="override evaluation.denoising_steps to measure the per-step cost",
    )
    args = parser.parse_args()

    from dreamwam.config import load_release_config
    from dreamwam.policy import build_policy

    config = load_release_config(args.config)
    # Build dense, then attach the sparse configuration.  The sparse path adds no checkpoint
    # parameters, so the same loaded model serves every variant and the comparison isolates
    # the attention policy rather than the weights.
    policy = build_policy(config, device=args.device)
    evaluation = policy.evaluation
    if args.denoising_steps is not None:
        # Same weights, same inputs, same sampler - only the number of steps changes, so
        # the difference is the cost of one step repeated, not a different model.
        evaluation["denoising_steps"] = int(args.denoising_steps)
    if args.sparse_json:
        policy.sparse_config = SparseConfig.from_mapping(json.loads(args.sparse_json))

    timer = EventTimer()
    phases = PhaseTimer()
    original = None
    if not args.no_instrument:
        original = instrument(policy, timer, phases)

    image_size = int(policy.image_size)
    images, state, instruction = build_inputs(args.seed, image_size)

    latencies: list[float] = []
    for index in range(args.warmup + args.reps):
        timer.reset()
        phases.reset()
        torch.cuda.synchronize()
        start = time.perf_counter()
        action = policy.predict_action(
            images=images,
            state=state,
            instruction=instruction,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if index >= args.warmup:
            latencies.append(elapsed)
    if not np.isfinite(np.asarray(action)).all():
        raise FloatingPointError("policy produced non-finite actions")

    attention_totals = timer.totals_ms()
    mean_latency = statistics.fmean(latencies)

    # Geometry, read from the real tensors rather than from the YAML.
    model = policy.model
    num_layers = int(model.config.num_layers)
    num_heads = int(model.config.num_heads)
    video_frames = int(evaluation["video_frames"])
    temporal = int(config.model.vae_temporal_downsample_factor)
    latent_frames = (video_frames - 1) // temporal + 1
    tokens_per_frame = (image_size // 16 // config.model.patch_size[1]) * (
        (2 * image_size) // 16 // config.model.patch_size[2]
    )
    video_tokens = latent_frames * tokens_per_frame
    action_tokens = int(evaluation["action_horizon"])
    steps = int(evaluation["denoising_steps"])

    self_attention_ms = attention_totals.get("self_attention", 0.0)
    cross_attention_ms = attention_totals.get("cross_attention", 0.0)
    phase_seconds = dict(phases.seconds)
    per_call_phases = {
        name: value / max(1, phases.counts.get(name, 1))
        for name, value in phase_seconds.items()
    }

    report = {
        "script": "scripts/sparse/profile_dense.py",
        "stage": "S1-dense-profiling",
        "status": "MEASURED",
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "dtype": str(policy.dtype),
        "config": str(args.config),
        "setting": config.setting,
        "sparse": policy.sparse_config.describe(),
        "checkpoint": str(config.paths.checkpoint),
        "geometry": {
            "image_size": image_size,
            "latent_frames": latent_frames,
            "tokens_per_frame": tokens_per_frame,
            "video_tokens": video_tokens,
            "action_tokens": action_tokens,
            "sequence_length": video_tokens + action_tokens,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "attn_head_dim": int(config.model.attn_head_dim),
            "denoising_steps": steps,
            "action_horizon": action_tokens,
            "replan_steps": int(evaluation.get("replan_steps", 0)),
        },
        "request_latency_seconds": {
            "samples": len(latencies),
            "mean": mean_latency,
            "p50": percentile(latencies, 0.5),
            "p95": percentile(latencies, 0.95),
            "min": min(latencies),
            "max": max(latencies),
        },
        "instrumented": not args.no_instrument,
        "attention_ms_per_request": {
            "self_attention": self_attention_ms,
            "cross_attention": cross_attention_ms,
            "self_attention_calls": timer.calls.get("self_attention", 0),
            "cross_attention_calls": timer.calls.get("cross_attention", 0),
            "self_attention_shapes_qkwd": list(timer.shapes.get("self_attention", ())),
            "cross_attention_shapes_qkwd": list(timer.shapes.get("cross_attention", ())),
        },
        "phase_seconds_per_request": per_call_phases,
        "phase_counts_per_request": dict(phases.counts),
        "vram_gib": {
            "allocated": torch.cuda.memory_allocated() / 2**30,
            "peak_allocated": torch.cuda.max_memory_allocated() / 2**30,
        },
    }

    expected_self_calls = num_layers * (steps if config.setting == "joint" else 1)
    report["attention_ms_per_request"]["expected_self_attention_calls"] = expected_self_calls
    if self_attention_ms > 0:
        report["shares"] = {
            "self_attention_fraction_of_request": self_attention_ms / 1000.0 / mean_latency,
            "cross_attention_fraction_of_request": cross_attention_ms / 1000.0 / mean_latency,
            "joint_denoise_fraction_of_request": per_call_phases.get("joint_denoise", 0.0)
            / mean_latency,
        }
        # Upper bound if VV sparsity were free and removed every self-attention FLOP.
        report["amdahl"] = {
            "vv_free_speedup_upper_bound": mean_latency
            / max(mean_latency - self_attention_ms / 1000.0, 1e-9),
        }

    if args.profile_kernels and not args.no_instrument:
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
            policy.predict_action(images=images, state=state, instruction=instruction)
        def device_ms(entry) -> float:
            # torch renamed cuda_time_total -> device_time_total; support both.
            for attribute in ("device_time_total", "cuda_time_total"):
                value = getattr(entry, attribute, None)
                if value is not None:
                    return float(value) / 1000.0
            return 0.0

        report["kernels"] = [
            {
                "name": entry.key,
                "cuda_time_ms": device_ms(entry),
                "calls": entry.count,
            }
            for entry in prof.key_averages()
        ]
        report["kernels"].sort(key=lambda item: -item["cuda_time_ms"])
        report["kernels"] = report["kernels"][:40]

    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")

    if original is not None:
        import dreamwam.layers as layers_module
        import dreamwam.mot as mot_module

        layers_module.scaled_dot_product_attention = original
        mot_module.scaled_dot_product_attention = original


if __name__ == "__main__":
    main()
