#!/usr/bin/env python3
"""Profile real checkpoint requests to explain cache costs, not claim speedups.

Collects one CPU/CUDA trace per variant after warmup. Full-request latency claims
come from benchmark_ffn_context_cache.py without profiler instrumentation. The
profiled actions must be bitwise the same as each variant's uninstrumented output.
No optimization, environment change or SR judgement is performed here.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np
import torch

from benchmark_ffn_context_cache import command, inputs_for, sha256
from dreamwam.config import load_release_config
from dreamwam.policy import build_policy
from dreamwam.sparse.action_guided_visual_token_cache import ActionGuidedVisualTokenCache
from dreamwam.sparse.visual_step_cache import VisualStepCache


@contextmanager
def annotate(owner, attribute, label):
    original = getattr(owner, attribute)
    def instrumented(*args, **kwargs):
        with torch.profiler.record_function(label):
            return original(*args, **kwargs)
    setattr(owner, attribute, instrumented)
    try:
        yield
    finally:
        setattr(owner, attribute, original)


def event_summary(event):
    return dict(name=event.key, calls=event.count,
                cpu_total_us=event.cpu_time_total, cpu_self_us=event.self_cpu_time_total,
                device_total_us=event.device_time_total, device_self_us=event.self_device_time_total)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dreamwam_joint.yaml")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    if args.warmup < 1:
        parser.error("warmup must be positive")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    release = load_release_config(args.config)
    inventory = lambda: command("nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,memory.used", "--format=csv")
    manifest = dict(status="RUNNING", start_utc=datetime.now(timezone.utc).isoformat(),
                    argv=sys.argv, pid=os.getpid(), cwd=str(Path.cwd()),
                    git=command("git", "rev-parse", "HEAD"),
                    dirty=command("git", "status", "--porcelain"),
                    python=platform.python_version(), torch=torch.__version__,
                    cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                    checkpoint_sha256=sha256(release.paths.checkpoint),
                    script_sha256=sha256(__file__), gpu_before=inventory(),
                    sr=None, speedup=None,
                    caveat="Instrumented single-request traces are diagnostic, not latency samples; nested stage totals overlap.")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    policy = build_policy(release, device="cuda")
    inputs, input_record = inputs_for(argparse.Namespace(inputs_npz=None), policy.image_size)
    images, state, instruction = inputs[0]
    manifest.update(inputs=input_record, input_id=0, evaluation=dict(policy.evaluation), dtype=str(policy.dtype))
    versions = {name: parameter._version for name, parameter in policy.model.named_parameters()}
    variants = dict(
        matched_dense=ActionGuidedVisualTokenCache(policy.model, keep_ratio=1, refresh_every=1),
        temporal_only=VisualStepCache(policy.model, refresh_every=5),
        guided_tokens=ActionGuidedVisualTokenCache(policy.model, keep_ratio=0.1, refresh_every=5, guidance_weight=1),
    )
    predict = lambda: policy.predict_action(images=images, state=state, instruction=instruction)
    try:
        native = predict()
        for name, cache in variants.items():
            with cache:
                for _ in range(args.warmup):
                    predict()
                expected = predict()
                if name == "matched_dense" and not np.array_equal(native, expected):
                    raise AssertionError("matched control is not bitwise native Dense")
                with ExitStack() as stack:
                    stages = (
                        (policy, "text_encoder", "stage.text"),
                        (policy, "_encode_first_frame", "stage.vae"),
                        (policy.model, "sample_action", "stage.sampling"),
                        (policy.model.video_expert, "pre_dit", "stage.video_pre_dit"),
                        (policy.model.video_expert, "post_dit", "stage.video_post_dit"),
                        (policy.model.video_scheduler, "step", "stage.video_scheduler"),
                        (policy.model.action_scheduler, "step", "stage.action_scheduler"),
                        (cache, "_refresh", "stage.refresh"),
                        (cache, "_reuse", "stage.reuse"),
                    )
                    for owner, attribute, label in stages:
                        stack.enter_context(annotate(owner, attribute, label))
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                           torch.profiler.ProfilerActivity.CUDA]) as profile:
                        with torch.profiler.record_function("stage.full_predict_action"):
                            actual = predict()
                            torch.cuda.synchronize()
                if not np.array_equal(expected, actual):
                    raise AssertionError(f"profiling changed actions: {name}")
                events = [event_summary(event) for event in profile.key_averages()]
                result = dict(variant=name, bitwise_uninstrumented_parity=True, cache=cache.last_stats,
                              stages=[event for event in events if event["name"].startswith("stage.")],
                              top_device_self=sorted(events, key=lambda row: row["device_self_us"], reverse=True)[:40],
                              top_cpu_self=sorted(events, key=lambda row: row["cpu_self_us"], reverse=True)[:40])
                profile.export_chrome_trace(str(out / f"{name}-trace.json"))
                (out / f"{name}-profile.json").write_text(json.dumps(result, indent=2) + "\n")
                print(json.dumps(dict(variant=name, stages=result["stages"], parity=True)), flush=True)
        if versions != {name: parameter._version for name, parameter in policy.model.named_parameters()}:
            raise AssertionError("model parameters changed during profiling")
        manifest.update(status="PROFILED", exit_code=0, end_utc=datetime.now(timezone.utc).isoformat(),
                        gpu_after=inventory(), unchanged_parameter_versions=True)
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    finally:
        policy.close()


if __name__ == "__main__":
    main()
