#!/usr/bin/env python3
"""A final-layer/final-step VV perturbation must leave final actions unchanged.

Joint attention computes video and action rows from the same incoming state.
Their post-attention paths are separate. With no later layer or sampling step,
changing only the final video-row mask has no causal path to the final action.
This control checks that claim with the real checkpoint and all 24 heads.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

from benchmark_ffn_context_cache import command, sha256
from collect_head_stage_impact import relative_l2
from dreamwam.config import load_release_config
from dreamwam.policy import build_policy
from dreamwam.sparse.head_stage_probe import HeadStageProbe


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/dreamwam_joint.yaml")
    parser.add_argument("--inputs",type=Path,required=True)
    parser.add_argument("--out-dir",type=Path,required=True)
    args=parser.parse_args()
    dataset=json.loads(args.inputs.read_text())
    if dataset["status"] != "CAPTURED" or len(dataset["inputs"]) != 3:
        parser.error("use all three frozen M1 inputs")
    args.out_dir.mkdir(parents=True,exist_ok=False)
    release=load_release_config(args.config)
    report=dict(status="RUNNING",start_utc=datetime.now(timezone.utc).isoformat(),
        argv=sys.argv,pid=os.getpid(),git=command("git","rev-parse","HEAD"),
        input_manifest_sha256=sha256(args.inputs),checkpoint_sha256=sha256(release.paths.checkpoint),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        sources={p:sha256(p) for p in [__file__,"dreamwam/sparse/head_stage_probe.py",
            "dreamwam/mot.py","dreamwam/model.py",args.config]},
        intervention="one head, last layer, final sampling step only; remove future VV pairs",
        sr=None,records=[])
    write=lambda:(args.out_dir/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    write()
    try:
        policy=build_policy(release,device="cuda")
        evaluation=dict(policy.evaluation)
        versions={k:p._version for k,p in policy.model.named_parameters()}
        if evaluation["denoising_steps"] != 10 or evaluation["action_horizon"] != 32:
            raise ValueError("retain released sampling protocol")
        mot=policy.model.mot
        for entry in dataset["inputs"]:
            source=args.inputs.parent/entry["path"]
            if sha256(source) != entry["sha256"]:raise ValueError("frozen input changed")
            with np.load(source,allow_pickle=False) as data:
                images={k:data[k].copy() for k in ["agentview","wrist"]}
                state,instruction=data["state"].copy(),str(data["instruction"].item())
            predict=lambda:policy.predict_action(images=images,state=state,instruction=instruction)
            native=predict()
            with HeadStageProbe(policy.model) as probe:
                baseline=predict()
                if not np.array_equal(native,baseline):raise AssertionError("full-budget native parity failed")
                video,raw=probe.last_video.clone(),probe.last_raw_action.clone()
                for head in range(mot.num_heads):
                    probe.configure()
                    original=mot._joint_self_attention

                    def final_only(**kwargs):
                        if kwargs["step_index"]==9 and kwargs["layer_index"]==0:
                            probe.configure(layer=mot.num_layers-1,head=head,stage=2,future_keep_ratio=0.0)
                        return original(**kwargs)

                    mot._joint_self_attention=final_only
                    try:actual=predict()
                    finally:mot._joint_self_attention=original
                    stats=dict(probe.stats)
                    if stats["target_steps"] != [9] or stats["target_calls"] != 1:
                        raise AssertionError("final-step control scope changed")
                    row=dict(input_id=entry["id"],layer=mot.num_layers-1,head=head,step=9,
                        actions_bitwise=bool(np.array_equal(actual,baseline)),
                        normalized_actions_bitwise=bool(torch.equal(probe.last_raw_action,raw)),
                        future_video_relative_l2=relative_l2(probe.last_video[:,:,1:],video[:,:,1:]),
                        conditioned_frame_bitwise=bool(torch.equal(probe.last_video[:,:,:1],video[:,:,:1])),
                        stats=stats)
                    report["records"].append(row);write()
                    if not all(row[k] for k in ["actions_bitwise","normalized_actions_bitwise","conditioned_frame_bitwise"]):
                        raise AssertionError("video-only intervention changed causally unaffected actions/conditioning")
                probe.configure()
                if not np.array_equal(predict(),baseline) or not torch.equal(probe.last_video,video):
                    raise AssertionError("Dense drifted after final-step controls")
        if evaluation != policy.evaluation or versions != {k:p._version for k,p in policy.model.named_parameters()}:
            raise AssertionError("weights/settings changed")
        report.update(status="VERIFIED",end_utc=datetime.now(timezone.utc).isoformat(),
            planned_controls=3*mot.num_heads,completed_controls=len(report["records"]),evaluation=evaluation)
        write()
        print(json.dumps({k:v for k,v in report.items() if k!="records"},indent=2))
    except BaseException as exc:
        report.update(status="ERROR",end_utc=datetime.now(timezone.utc).isoformat(),error=repr(exc));write()
        raise


if __name__ == "__main__":
    main()
