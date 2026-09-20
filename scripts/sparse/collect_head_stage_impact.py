#!/usr/bin/env python3
"""Offline single-head/stage interventions continued to final actions and video.

Every label is paired on exactly the same captured real observation and seed.
This is calibration, not a speed or success-rate measurement. Full-budget
controls must be bitwise native Dense before collecting any labels.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

from benchmark_ffn_context_cache import command, sha256
from collect_action_impact import components
from dreamwam.config import load_release_config
from dreamwam.policy import build_policy
from dreamwam.sparse.head_stage_probe import HeadStageProbe, stage_bounds


def relative_l2(actual, expected):
    delta = actual.double() - expected.double()
    return float(torch.linalg.vector_norm(delta) /
                 torch.linalg.vector_norm(expected.double()).clamp_min(1e-12))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dreamwam_joint.yaml")
    parser.add_argument("--inputs", required=True, help="captured input manifest.json")
    parser.add_argument("--future-keep-ratio", type=float, default=0.0)
    parser.add_argument("--layers", type=int, nargs="+", default=None)
    parser.add_argument("--heads", type=int, nargs="+", default=None)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    if not 0 <= args.future_keep_ratio < 1:
        parser.error("calibration requires a restricted future-key budget in [0,1)")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    source = Path(args.inputs)
    dataset = json.loads(source.read_text())
    if dataset["status"] != "CAPTURED" or len(dataset["inputs"]) < 2:
        parser.error("at least two completely captured real inputs are required")
    release = load_release_config(args.config)
    manifest = dict(status="RUNNING", start_utc=datetime.now(timezone.utc).isoformat(),
        argv=sys.argv, pid=os.getpid(), git=command("git", "rev-parse", "HEAD"),
        dirty=command("git", "status", "--porcelain"), python=platform.python_version(),
        torch=torch.__version__, cuda=torch.version.cuda,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        checkpoint_sha256=sha256(release.paths.checkpoint),
        input_manifest=str(source), input_manifest_sha256=sha256(source),
        inputs=dataset["inputs"], stages=stage_bounds(10),
        future_keep_ratio=args.future_keep_ratio, mask_family="one-head future-query VV future-key restriction; conditioning and action rows preserved",
        sources={name:sha256(name) for name in ["dreamwam/sparse/head_stage_probe.py",
            "scripts/sparse/collect_head_stage_impact.py", "dreamwam/mot.py", "dreamwam/layers.py", args.config]},
        environment="existing Python/torch reused; no install or sync",
        interpretation="offline perturbation sensitivity, not certified SR safety or speedup",
        controls=[], completed_interventions=0, sr=None)
    write = lambda: (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write()
    try:
        started = time.perf_counter()
        policy = build_policy(release, device="cuda")
        evaluation = dict(policy.evaluation)
        if evaluation["denoising_steps"] != 10 or evaluation["action_horizon"] != 32:
            raise ValueError("retain released ten steps and horizon 32")
        versions = {k:p._version for k,p in policy.model.named_parameters()}
        layers = list(range(policy.model.mot.num_layers)) if args.layers is None else args.layers
        heads = list(range(policy.model.mot.num_heads)) if args.heads is None else args.heads
        if len(set(layers)) != len(layers) or len(set(heads)) != len(heads):
            raise ValueError("duplicate layer/head")
        for layer in layers:
            for head in heads:
                HeadStageProbe(policy.model).configure(layer=layer, head=head, stage=0)
        manifest.update(model_load_seconds=time.perf_counter()-started, evaluation=evaluation,
            dtype=str(policy.dtype), layers=layers, heads=heads,
            planned_interventions=len(layers)*len(heads)*3*len(dataset["inputs"]))
        write()
        with (out / "interventions.jsonl").open("w", buffering=1) as raw:
            for input_index, entry in enumerate(dataset["inputs"]):
                path = source.parent / entry["path"]
                if sha256(path) != entry["sha256"]:
                    raise ValueError("captured input changed")
                with np.load(path, allow_pickle=False) as data:
                    images = {k:data[k].copy() for k in ["agentview", "wrist"]}
                    state, instruction = data["state"].copy(), str(data["instruction"].item())
                predict = lambda: policy.predict_action(images=images,state=state,instruction=instruction)
                native = predict()
                if not np.isfinite(native).all() or not np.array_equal(native,predict()):
                    raise AssertionError("native fixed-input actions are not bitwise repeatable")
                with HeadStageProbe(policy.model) as probe:
                    baseline = predict()
                    if not np.array_equal(native, baseline):
                        raise AssertionError("expanded-mask full-budget control is not bitwise native Dense")
                    video = probe.last_video.clone()
                    action_raw = probe.last_raw_action.clone()
                    np.savez_compressed(out/f"reference-{entry['id']}.npz", actions=baseline,
                        normalized_actions=action_raw.float().cpu().numpy(), video_latents=video.float().cpu().numpy())
                    control=dict(input_id=entry["id"], native_repeat_bitwise=True,
                        full_budget_native_bitwise=True, reference_stats=dict(probe.stats))
                    manifest["controls"].append(control)
                    write()
                    for layer in layers:
                        for head in heads:
                            for stage,(lo,hi) in enumerate(stage_bounds(10)):
                                probe.configure(layer=layer,head=head,stage=stage,
                                                future_keep_ratio=args.future_keep_ratio)
                                started=time.perf_counter()
                                action=predict()
                                if not np.isfinite(action).all() or not torch.isfinite(probe.last_video).all():
                                    raise FloatingPointError("non-finite intervention output")
                                stats=dict(probe.stats)
                                if (stats["target_steps"] != list(range(lo,hi)) or
                                    stats["attention_calls"] != policy.model.mot.num_layers*10 or
                                    stats["video_scheduler_steps"] != 10):
                                    raise AssertionError("intervention scope or sampling step count changed")
                                row=dict(input_id=entry["id"],input_index=input_index,layer=layer,
                                    head=head,stage=stage,step_range=[lo,hi],
                                    action=components(baseline,action),
                                    normalized_action_relative_l2=relative_l2(probe.last_raw_action,action_raw),
                                    future_video_relative_l2=relative_l2(probe.last_video[:,:,1:],video[:,:,1:]),
                                    conditioned_frame_bitwise=bool(torch.equal(probe.last_video[:,:,:1],video[:,:,:1])),
                                    actions=action.tolist(),stats=stats,
                                    diagnostic_wall_seconds=time.perf_counter()-started)
                                if not row["conditioned_frame_bitwise"]:
                                    raise AssertionError("conditioned frame changed")
                                raw.write(json.dumps(row)+"\n")
                                manifest["completed_interventions"]+=1
                        # Dense drift control after each whole layer; never silently correct drift.
                        probe.configure()
                        if not np.array_equal(predict(),baseline) or not torch.equal(probe.last_video,video):
                            raise AssertionError("matched Dense reference drifted during calibration")
                        write()
                        print(f"{entry['id']} layer {layer}: {manifest['completed_interventions']}/{manifest['planned_interventions']} interventions",flush=True)
        if versions != {k:p._version for k,p in policy.model.named_parameters()} or evaluation != policy.evaluation:
            raise AssertionError("weights or evaluation settings changed")
        manifest.update(status="MEASURED",end_utc=datetime.now(timezone.utc).isoformat())
        write()
        print(json.dumps({k:manifest[k] for k in ['status','planned_interventions','completed_interventions','end_utc']}),flush=True)
    except BaseException as exc:
        manifest.update(status="ERROR",error=repr(exc),end_utc=datetime.now(timezone.utc).isoformat())
        write()
        raise


if __name__ == "__main__":
    main()
