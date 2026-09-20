#!/usr/bin/env python3
"""Finite native Dense Q/K/V capture, not a timing benchmark or SR evaluation."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from dreamwam.config import load_release_config
from dreamwam.policy import build_policy
from dreamwam.sparse.profile.admission import admit_profile
from dreamwam.sparse.profile.archive import RawArchive, sha256
from dreamwam.sparse.profile.capture import DenseProfile


def stamp():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("docs/implementation/dido-sparse-profile/experiment-plan.json"))
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--config", default="configs/dreamwam_joint.yaml")
    args = parser.parse_args()
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        parser.error("require clean committed source in an immutable H100 run worktree")
    if not str(Path.cwd()).startswith("/root/wenbiao_zhao/dreamwam-sr/.trees/dido-run-"):
        parser.error("run on H100 in a detached dido-run-* worktree")
    plan = json.loads(args.plan.read_text())
    sampling = plan["development"]
    if sha256(args.inputs) != sampling["profile_inputs_sha256"]:
        parser.error("development input manifest differs from the frozen plan")
    dataset = json.loads(args.inputs.read_text())
    if dataset.get("status") != "CAPTURED" or dataset.get("split_role") != "development":
        parser.error("require explicitly labelled captured development data")
    entries = dataset["inputs"]
    if len(entries) != sampling["input_count"] or 2 * len(entries) > sampling["max_profile_predict_calls"]:
        parser.error("input count exceeds or differs from the declared call budget")
    if len({item["id"] for item in entries}) != len(entries):
        parser.error("duplicate input identities")
    for item in entries:
        if sha256(args.inputs.parent / item["path"]) != item["sha256"]:
            parser.error("raw observation hash mismatch")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible.startswith("GPU-") or "," in visible:
        parser.error("select exactly one explicitly admitted H100 GPU UUID")
    admission = admit_profile(visible, authorized=tuple(plan["authorized_h100_gpu_indices"]))
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        parser.error("CUDA is required; CPU checkpoint fallback is forbidden")
    release = load_release_config(args.config)
    if sha256(release.paths.checkpoint) != plan["checkpoint_sha256"]:
        parser.error("checkpoint differs from the frozen reference")
    if release.evaluation["denoising_steps"] != 10 or release.evaluation["action_horizon"] != 32:
        parser.error("sampling protocol changed")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    archive = RawArchive(args.out_dir / "raw", max_bytes=sampling["max_profile_raw_bytes"])
    report = dict(status="RUNNING", start_utc=stamp(), source_commit=source, argv=sys.argv,
        pid=os.getpid(), checkpoint_sha256=plan["checkpoint_sha256"],
        config_sha256=sha256(args.config), plan_sha256=sha256(args.plan),
        input_manifest_sha256=sha256(args.inputs), sampling=sampling, admission=admission,
        python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
        split_role="development", planned_calls=2 * len(entries), completed_calls=0,
        inputs=[], sr=None, speedup=None,
        limitations=["sampled heads and layers, not full coverage of all heads/layers",
            "float32 analytical softmax is descriptive, native fused output is preserved",
            "value_action is our proxy, not DIDO action V-attribution",
            "spatial footprint camera labels are not exclusive receptive fields",
            "self-captured observations, not author data or semantic annotations"])

    def write_report():
        path = args.out_dir / "report.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        tmp.replace(path)

    write_report()
    policy = None
    try:
        # Admission is refreshed immediately before model loading, after hashing.
        report["load_admission"] = admit_profile(visible, authorized=tuple(plan["authorized_h100_gpu_indices"]))
        policy = build_policy(release, device="cuda", prompt_cache={"capacity": 8})
        versions = tuple(p._version for p in policy.model.parameters())
        for item in entries:
            with np.load(args.inputs.parent / item["path"], allow_pickle=False) as raw:
                observation = dict(images={k: raw[k].copy() for k in ("agentview", "wrist")},
                    state=raw["state"].copy(), instruction=str(raw["instruction"].item()))
            expected = policy.predict_action(**observation)
            report["completed_calls"] += 1
            sink = lambda kind, metadata, arrays: archive.write(item["id"], kind, metadata, arrays)
            with DenseProfile(policy.model, steps=sampling["capture_steps"],
                    layers=sampling["capture_layers"], heads=sampling["capture_heads"], sink=sink,
                    max_records=len(sampling["capture_steps"]) * len(sampling["capture_layers"]),
                    max_bytes=sampling["max_profile_raw_bytes"] - archive.raw_bytes) as profile:
                actual = policy.predict_action(**observation)
                report["completed_calls"] += 1
                if (profile.grid.frames, profile.grid.height, profile.grid.width) != (3, 7, 14):
                    raise AssertionError("real checkpoint token grid differs from audited geometry")
                if not np.array_equal(actual, expected):
                    raise AssertionError("raw profiling changed native Dense actions")
                archive.write(item["id"], "actions", dict(request=profile.request),
                              dict(native=expected, instrumented=actual))
                report["inputs"].append(dict(input_id=item["id"], sha256=item["sha256"],
                    action_parity=True, grid=[3, 7, 14], stats=dict(profile.stats)))
            if tuple(p._version for p in policy.model.parameters()) != versions:
                raise AssertionError("model parameters changed")
            write_report()
            print(json.dumps(dict(input_id=item["id"], completed_calls=report["completed_calls"],
                                  raw_bytes=archive.raw_bytes)), flush=True)
        report["status"] = "COMPLETE"
        report["exit_code"] = 0
    except BaseException as exc:
        report.update(status="ERROR", exit_code=1, error=repr(exc))
        raise
    finally:
        if policy is not None:
            policy.close()
        report.update(end_utc=stamp(), raw_bytes=archive.raw_bytes, artifacts=len(archive.records),
                      raw_index_sha256=sha256(archive.root / "records.jsonl")
                      if (archive.root / "records.jsonl").exists() else None)
        write_report()


if __name__ == "__main__":
    main()
