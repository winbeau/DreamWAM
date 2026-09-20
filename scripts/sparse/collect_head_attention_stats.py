#!/usr/bin/env python3
"""Observe native Dense AV/VV statistics on the frozen M1 replay inputs.

The original fused attention produces every model output. Separate float32
Q/K softmax calculations supply descriptive statistics only; their costs are
not an inference benchmark and their magnitudes are not sensitivity labels.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import sys

import numpy as np
import torch

from benchmark_ffn_context_cache import command, sha256
from dreamwam.config import load_release_config
from dreamwam.policy import build_policy
from dreamwam.sparse.head_stage_probe import HeadStageProbe, stage_bounds


def attention_statistics(kwargs, heads):
    """Return one vector per head, respecting the full native joint mask."""
    video, action = kwargs["video_io"], kwargs["action_io"]
    query = torch.cat([video[0], action[0]], dim=1)
    key = torch.cat([video[1], action[1]], dim=1)
    batch, length, width = query.shape
    if batch != 1 or width % heads:
        raise ValueError("the fixed replay expects batch one and complete heads")
    dim = width // heads
    query = query.float().reshape(batch, length, heads, dim).transpose(1, 2)
    key = key.float().reshape(batch, length, heads, dim).transpose(1, 2)
    logits = (query @ key.transpose(-1, -2)) / math.sqrt(dim)
    mask = kwargs["attention_mask"]
    if mask.ndim != 2 or mask.dtype != torch.bool:
        raise ValueError("statistics require the native two-dimensional joint mask")
    probabilities = logits.masked_fill(~mask, -torch.inf).softmax(dim=-1)[0]
    nv, first = kwargs["video_length"], kwargs["tokens_per_frame"]
    av = probabilities[:, nv:, :].mean(dim=1)
    vv = probabilities[:, first:nv, :].mean(dim=1)
    av_future, vv_future = av[:, first:nv], vv[:, first:nv]
    normalize = lambda value: value / value.sum(dim=-1, keepdim=True).clamp_min(1e-30)
    av_distribution, vv_distribution = normalize(av_future), normalize(vv_future)
    entropy = lambda value: -(value * value.clamp_min(1e-30).log()).sum(dim=-1) / math.log(nv-first)
    cosine = (av_distribution * vv_distribution).sum(dim=-1) / (
        av_distribution.square().sum(dim=-1).sqrt() *
        vv_distribution.square().sum(dim=-1).sqrt()).clamp_min(1e-30)
    return dict(
        av_conditioned_mass=av[:, :first].sum(dim=-1),
        av_future_mass=av_future.sum(dim=-1),
        av_action_mass=av[:, nv:].sum(dim=-1),
        vv_conditioned_mass=vv[:, :first].sum(dim=-1),
        vv_future_mass=vv_future.sum(dim=-1),
        vv_action_mass=vv[:, nv:].sum(dim=-1),
        av_future_aggregate_entropy=entropy(av_distribution),
        vv_future_aggregate_entropy=entropy(vv_distribution),
        av_vv_future_aggregate_cosine=cosine,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dreamwam_joint.yaml")
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    dataset = json.loads(args.inputs.read_text())
    if dataset["status"] != "CAPTURED" or len(dataset["inputs"]) != 3:
        parser.error("use the three frozen M1 inputs")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    release = load_release_config(args.config)
    manifest = dict(status="RUNNING", start_utc=datetime.now(timezone.utc).isoformat(),
        argv=sys.argv, git=command("git", "rev-parse", "HEAD"),
        dirty=command("git", "status", "--porcelain"), pid=os.getpid(),
        python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        checkpoint_sha256=sha256(release.paths.checkpoint),
        input_manifest_sha256=sha256(args.inputs), inputs=dataset["inputs"],
        sources={name:sha256(name) for name in [__file__, "dreamwam/sparse/head_stage_probe.py",
            "dreamwam/mot.py", "dreamwam/layers.py", args.config]},
        controls=[], stages=stage_bounds(10), sr=None,
        interpretation="observational joint-softmax statistics; not causal types, SR or speed",
        definitions={
            "av": "native action queries, full joint [V,A] normalization; mean over action queries",
            "vv": "future video queries, native mask; mean over future video queries",
            "entropy": "entropy of the normalized aggregate future-key distribution, divided by log(number of future keys)",
            "cosine": "cosine between AV and VV aggregate future-key distributions",
            "stage": "arithmetic mean of step statistics in the fixed 3/4/3 step ranges",
        })
    write = lambda: (args.out_dir/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    write()
    try:
        policy = build_policy(release,device="cuda")
        evaluation = dict(policy.evaluation)
        if evaluation["denoising_steps"] != 10 or evaluation["action_horizon"] != 32:
            raise ValueError("retain the frozen sampling protocol")
        versions = {k:p._version for k,p in policy.model.named_parameters()}
        mot = policy.model.mot
        all_rows = []
        for entry in dataset["inputs"]:
            source = args.inputs.parent/entry["path"]
            if sha256(source) != entry["sha256"]:
                raise ValueError("frozen replay input changed")
            with np.load(source,allow_pickle=False) as data:
                images = {k:data[k].copy() for k in ["agentview","wrist"]}
                state, instruction = data["state"].copy(),str(data["instruction"].item())
            predict = lambda: policy.predict_action(images=images,state=state,instruction=instruction)
            native = predict()
            observations = []
            with HeadStageProbe(policy.model) as probe:
                if not np.array_equal(native,predict()):
                    raise AssertionError("full-budget control is not native Dense")
                video, raw = probe.last_video.clone(),probe.last_raw_action.clone()
                original = mot._joint_self_attention

                def observe(**kwargs):
                    result = original(**kwargs)
                    # kwargs still has the native mask: the full-budget probe
                    # expands its private copy only inside original(...).
                    values = attention_statistics(kwargs,mot.num_heads)
                    if any(not torch.isfinite(value).all() for value in values.values()):
                        raise FloatingPointError("non-finite attention statistic")
                    vectors = {k:v.detach().cpu().tolist() for k,v in values.items()}
                    step = kwargs["step_index"]
                    stage = next(i for i,(lo,hi) in enumerate(stage_bounds(10)) if lo <= step < hi)
                    for head in range(mot.num_heads):
                        observations.append(dict(input_id=entry["id"],layer=kwargs["layer_index"],
                            head=head,step=step,stage=stage,**{k:v[head] for k,v in vectors.items()}))
                    return result

                mot._joint_self_attention = observe
                try:
                    instrumented = predict()
                finally:
                    mot._joint_self_attention = original
                if (not np.array_equal(native,instrumented) or not torch.equal(video,probe.last_video)
                    or not torch.equal(raw,probe.last_raw_action)):
                    raise AssertionError("observational profiling changed Dense outputs")
            expected = mot.num_layers * mot.num_heads * 10
            identities = {(r["layer"],r["head"],r["step"]) for r in observations}
            if len(observations) != expected or len(identities) != expected:
                raise AssertionError("incomplete or duplicate statistics coverage")
            all_rows.extend(observations)
            manifest["controls"].append(dict(input_id=entry["id"],native_action_bitwise=True,
                full_budget_video_bitwise=True,full_budget_raw_action_bitwise=True,step_head_records=expected))
            write()
        if evaluation != policy.evaluation or versions != {k:p._version for k,p in policy.model.named_parameters()}:
            raise AssertionError("weights or settings changed")
        with (args.out_dir/"step-head-statistics.jsonl").open("w") as handle:
            for row in all_rows:handle.write(json.dumps(row)+"\n")
        metrics = list(all_rows[0].keys())[5:]
        stage_rows = []
        for entry in dataset["inputs"]:
            selected = [r for r in all_rows if r["input_id"] == entry["id"]]
            groups = {}
            for row in selected:groups.setdefault((row["layer"],row["head"],row["stage"]),[]).append(row)
            for (layer,head,stage),rows in sorted(groups.items()):
                stage_rows.append(dict(input_id=entry["id"],layer=layer,head=head,stage=stage,
                    **{k:float(np.mean([r[k] for r in rows])) for k in metrics}))
        (args.out_dir/"head-stage-attention.json").write_text(json.dumps(stage_rows,indent=2)+"\n")
        manifest.update(status="MEASURED",end_utc=datetime.now(timezone.utc).isoformat(),
            evaluation=evaluation,step_head_records=len(all_rows),head_stage_records=len(stage_rows),
            artifacts={name:sha256(args.out_dir/name) for name in ["step-head-statistics.jsonl","head-stage-attention.json"]})
        write()
        print(json.dumps(manifest,indent=2))
    except BaseException as exc:
        manifest.update(status="ERROR",error=repr(exc),end_utc=datetime.now(timezone.utc).isoformat())
        write()
        raise


if __name__ == "__main__":
    main()
