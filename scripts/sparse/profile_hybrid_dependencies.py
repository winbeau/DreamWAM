#!/usr/bin/env python3
"""Offline interventions: action sensitivity versus future-latent sensitivity.

Full dense inference, never a speed benchmark or an SR measurement. First-layer
scores are only hypotheses; same-size key removal measures their downstream effect.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from benchmark_ffn_context_cache import command, sha256
from benchmark_hybrid_schedules import diagnostics, write_json
from dreamwam.config import load_release_config
from dreamwam.policy import build_policy
from dreamwam.sparse.hybrid.profiling import DependencyAudit, route_stability
from dreamwam.sparse.hybrid.search import parse_indices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--config", default="configs/dreamwam_joint.yaml")
    parser.add_argument("--steps", default="1,5,9")
    parser.add_argument("--methods", default="uniform,action,bottom_action,visual_context,action_context")
    parser.add_argument("--remove-ratio", type=float, default=0.1)
    parser.add_argument("--scope", choices=("all", "future"), default="all")
    parser.add_argument("--group-count", type=int, default=0,
                        help="also remove each disjoint within-frame index group; 0 disables")
    args = parser.parse_args()
    steps, methods = parse_indices(args.steps), args.methods.split(",")
    release = load_release_config(args.config)
    if not steps or any(step >= release.evaluation["denoising_steps"] for step in steps):
        parser.error("intervention steps must be inside the unchanged sampler")
    if len(methods) != len(set(methods)) or any(m not in ("uniform", "action", "bottom_action", "visual_context", "action_context") for m in methods):
        parser.error("unsupported/duplicate intervention methods")
    if not 0 < args.remove_ratio < 1:
        parser.error("remove-ratio must be in (0, 1)")
    if args.group_count == 1 or args.group_count < 0:
        parser.error("group-count must be zero or at least two")
    interventions = [(method, None) for method in methods] + [("group", i) for i in range(args.group_count)]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        parser.error("select one explicitly authorized CUDA device")
    source = command("git", "rev-parse", "HEAD")
    if source["exit_code"] or command("git", "status", "--porcelain")["stdout"]:
        parser.error("require clean committed source")
    captured = json.loads(args.inputs.read_text())
    if captured.get("status") != "CAPTURED" or not captured.get("inputs"):
        parser.error("require hash-verified real captured inputs")
    observations = {}
    for item in captured["inputs"]:
        path = args.inputs.parent / item["path"]
        if sha256(path) != item["sha256"] or item["id"] in observations:
            parser.error("input hash mismatch or duplicate identity")
        with np.load(path, allow_pickle=False) as data:
            observations[item["id"]] = dict(images={k: data[k].copy() for k in ("agentview", "wrist")},
                state=data["state"].copy(), instruction=str(data["instruction"].item()))
    args.out_dir.mkdir(parents=True, exist_ok=False)
    report = dict(status="RUNNING", start_utc=datetime.now(timezone.utc).isoformat(),
        argv=sys.argv, source=source, config_sha256=sha256(args.config),
        inputs_sha256=sha256(args.inputs), checkpoint_sha256=sha256(release.paths.checkpoint),
        hardware=command("nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv"),
        gpu=visible, torch=torch.__version__, cuda=torch.version.cuda, shared_host=True,
        split="exposed debug inputs; not held-out", scope=args.scope, group_count=args.group_count,
        planned=len(observations) * len(steps) * len(interventions),
        completed=0, baselines={}, interventions=[], sr=None,
        limitations=["first-layer attention proxy is not a causal oracle",
            "future-latent distance to Dense is not video quality or real-world prediction error",
            "key removal affects AV and VV jointly; not equivalent to token recompute pruning",
            "instrumented dense execution is not a timing benchmark"])
    write_json(args.out_dir / "report.json", report)
    policy = None
    try:
        policy = build_policy(release, device="cuda", prompt_cache={"capacity": 8})
        versions = tuple(p._version for p in policy.model.parameters())
        for name, observation in observations.items():
            with DependencyAudit(policy.model, collect=True, ratio=args.remove_ratio, scope=args.scope) as audit:
                dense = policy.predict_action(**observation)
            video = audit.future_latents.cpu().float().numpy()
            np.savez_compressed(args.out_dir / (name + "-dense.npz"), action=dense, future_latents=video)
            # The diagnostic wrapper cannot silently modify baseline actions.
            if not np.array_equal(policy.predict_action(**observation), dense):
                raise AssertionError("diagnostic instrumentation changed Dense actions")
            report["baselines"][name] = dict(records=audit.records, stability=route_stability(audit.records))
            for step in steps:
                for method, group_index in interventions:
                    with DependencyAudit(policy.model, remove_step=step, method=method,
                                         ratio=args.remove_ratio, scope=args.scope,
                                         group_index=group_index, group_count=args.group_count) as intervention:
                        actual = policy.predict_action(**observation)
                    if intervention.modified_layers != policy.model.mot.num_layers:
                        raise AssertionError("intervention did not cover exactly the requested layer-step")
                    actual_video = intervention.future_latents.cpu().float().numpy()
                    label = method if group_index is None else f"group_{group_index:02d}"
                    output = args.out_dir / f"{name}-{step:02d}-{label}.npz"
                    np.savez_compressed(output, action=actual, future_latents=actual_video)
                    report["interventions"].append(dict(input_id=name, step=step, method=label,
                        group_index=group_index, scope=args.scope,
                        removed=intervention.removed.cpu().tolist(), layers=intervention.modified_layers,
                        action=diagnostics(actual, dense),
                        executed_prefix=diagnostics(actual[:release.evaluation["replan_steps"]], dense[:release.evaluation["replan_steps"]]),
                        future_latent_relative_l2=float(np.linalg.norm(actual_video.astype(np.float64) - video) /
                                                       max(np.linalg.norm(video.astype(np.float64)), 1e-12)),
                        artifact=output.name, sha256=sha256(output)))
                    report["completed"] += 1
                    write_json(args.out_dir / "report.json", report)
                print(json.dumps(dict(input_id=name, step=step, completed=report["completed"])), flush=True)
        if tuple(p._version for p in policy.model.parameters()) != versions:
            raise AssertionError("profiling modified model weights")
        report["status"] = "COMPLETE"
    except BaseException as exc:
        report.update(status="ERROR", error=repr(exc))
        raise
    finally:
        if policy is not None:
            policy.close()
        report["end_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(args.out_dir / "report.json", report)


if __name__ == "__main__":
    main()
