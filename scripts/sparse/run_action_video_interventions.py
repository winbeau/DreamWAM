#!/usr/bin/env python3
"""At most 49 frozen diagnostic predictions, no rollout or timing benchmark."""

import argparse
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from dreamwam.config import load_release_config
from dreamwam.policy import build_policy
from dreamwam.sparse.profile.admission import admit_profile
from dreamwam.sparse.profile.analysis import index_capture, load_record, stamp, write_json
from dreamwam.sparse.profile.archive import RawArchive, sha256
from dreamwam.sparse.profile.intervention import KeyIntervention
from dreamwam.sparse.profile.study import output_diagnostics, prepare_cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("docs/implementation/dido-sparse-profile/experiment-plan.json"))
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--config", default="configs/dreamwam_joint.yaml")
    args = parser.parse_args()
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if (subprocess.check_output(["git", "status", "--porcelain"], text=True).strip() or
        not str(Path.cwd()).startswith("/root/wenbiao_zhao/dreamwam-sr/.trees/dido-run-")):
        parser.error("require clean immutable detached H100 source")
    plan = json.loads(args.plan.read_text())
    frozen = prepare_cases(plan, args.capture, args.analysis)
    capture, indexed = index_capture(args.capture)
    inputs_path = Path(capture["sampling"]["profile_inputs"])
    entries = {entry["id"]: entry for entry in json.loads(inputs_path.read_text())["inputs"]}
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible.startswith("GPU-") or "," in visible:
        parser.error("select exactly one admitted H100 UUID")
    admission_options = dict(authorized=tuple(plan["authorized_h100_gpu_indices"]),
                             share_gpu5=plan.get("allow_explicit_gpu5_sharing", False))
    admission = admit_profile(visible, **admission_options)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        parser.error("CUDA checkpoint execution is required")
    release = load_release_config(args.config)
    if sha256(release.paths.checkpoint) != frozen["checkpoint_sha256"]:
        parser.error("checkpoint changed")
    if (release.evaluation["denoising_steps"] != 10 or release.evaluation["action_horizon"] != 32 or
        release.evaluation["replan_steps"] != 10):
        parser.error("sampling/evaluation protocol changed")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    frozen.update(source_commit=source, plan_sha256=sha256(args.plan), start_utc=stamp())
    write_json(args.out_dir / "frozen-cases.json", frozen)
    archive = RawArchive(args.out_dir / "raw", max_bytes=plan["diagnostic_study"]["max_raw_bytes"])
    report = dict(status="RUNNING", start_utc=stamp(), argv=sys.argv, source_commit=source,
        checkpoint_sha256=frozen["checkpoint_sha256"], config_sha256=sha256(args.config),
        plan_sha256=sha256(args.plan), frozen_cases_sha256=sha256(args.out_dir / "frozen-cases.json"),
        pid=os.getpid(), python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
        input_kind=frozen["input_kind"], split_role="development", admission=admission,
        planned_calls=frozen["planned_calls"], attempted_calls=0, completed_calls=0,
        references=[], cases=[], sr=None, speedup=None,
        limitations=["two exposed development observations; this is diagnostic screening, not independent trials",
            "frozen Dense-derived indices are offline counterfactuals, not deployable teacher inputs",
            "delete, replacement and recompute have different count semantics; compare selectors within operation/scope",
            "all dense projection and FFN work remains; no speed claim",
            "current-frame video-time scores are tied at zero; no semantic object/interaction labels"])

    def save():
        report.update(raw_bytes=archive.raw_bytes, artifacts=len(archive.records))
        write_json(args.out_dir / "report.json", report)

    def predict(policy, observation, targets, operation, scope):
        if report["attempted_calls"] >= frozen["planned_calls"]:
            raise RuntimeError("predeclared policy-call cap reached")
        report["attempted_calls"] += 1
        save()
        with KeyIntervention(policy.model, targets, operation=operation, scope=scope) as probe:
            action = policy.predict_action(**observation)
            arrays = dict(action=action, raw_action=probe.raw_action.detach().float().cpu().numpy(),
                          video_latents=probe.last_video.detach().float().cpu().numpy())
            records = list(probe.records)
        report["completed_calls"] += 1
        return arrays, records

    policy = None
    save()
    try:
        report["load_admission"] = admit_profile(visible, **admission_options)
        policy = build_policy(release, device="cuda", prompt_cache={"capacity": 8})
        versions = tuple(parameter._version for parameter in policy.model.parameters())
        generator = random.Random(42)
        for name in frozen["input_ids"]:
            entry = entries[name]
            with np.load(inputs_path.parent / entry["path"], allow_pickle=False) as raw:
                observation = dict(images={k: raw[k].copy() for k in ("agentview", "wrist")},
                    state=raw["state"].copy(), instruction=str(raw["instruction"].item()))
            baseline, _ = predict(policy, observation, {}, "delete", "joint")
            original = load_record(args.capture / "raw", indexed[name, "reference", None, None])
            original["action"] = load_record(args.capture / "raw", indexed[name, "actions", None, None])["native"]
            if any(not np.array_equal(baseline[key], original[key]) for key in baseline):
                raise AssertionError("fresh native control differs from the verified raw capture")
            row = archive.write(name, "reference", dict(request=1), baseline)
            report["references"].append(dict(input_id=name, raw_and_executable_parity=True, artifact=row["path"]))
            cases = [case for case in frozen["cases"] if case["input_id"] == name]
            generator.shuffle(cases)
            for case in cases:
                actual, records = predict(policy, observation, {(case["step"], case["layer"]): case["indices"]},
                                          case["operation"], case["scope"])
                if len(records) != 1 or records[0]["indices"] != case["indices"]:
                    raise AssertionError("actual intervention differs from the frozen single-cell target")
                if case["scope"] == "AV" and not np.array_equal(actual["video_latents"], baseline["video_latents"]):
                    raise AssertionError("AV-only intervention unexpectedly changed the video path")
                if case["scope"] == "VV" and case["step"] == 9 and case["layer"] == 29:
                    if not np.array_equal(actual["raw_action"], baseline["raw_action"]):
                        raise AssertionError("last-layer last-step VV intervention changed same-step actions")
                row = archive.write(name, case["case_id"], dict(request=1, step=case["step"], layer=case["layer"]), actual)
                report["cases"].append(dict(**case, execution_order=len(report["cases"]), actual_records=records,
                    diagnostics=output_diagnostics(actual, baseline), artifact=row["path"], sha256=row["sha256"]))
                save()
                print(json.dumps(dict(case_id=case["case_id"], completed_calls=report["completed_calls"])), flush=True)
        if report["completed_calls"] != frozen["planned_calls"] or len(report["cases"]) != len(frozen["cases"]):
            raise AssertionError("diagnostic coverage incomplete")
        if tuple(parameter._version for parameter in policy.model.parameters()) != versions:
            raise AssertionError("checkpoint parameters changed")
        report.update(status="COMPLETE", exit_code=0)
    except BaseException as exc:
        report.update(status="ERROR", exit_code=1, error=repr(exc))
        raise
    finally:
        if policy is not None:
            report["peak_allocated_gpu_bytes"] = torch.cuda.max_memory_allocated()
            report["peak_reserved_gpu_bytes"] = torch.cuda.max_memory_reserved()
            policy.close()
        report.update(end_utc=stamp(), raw_index_sha256=sha256(archive.root / "records.jsonl")
            if (archive.root / "records.jsonl").exists() else None)
        save()


if __name__ == "__main__":
    main()
