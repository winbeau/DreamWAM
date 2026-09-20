#!/usr/bin/env python3
"""Summarize a completed M1 sweep into relative sensitivity classes and plots.

The first two captured inputs define quartile thresholds; the third is a
separate cross-input check. Labels describe this perturbation/dataset only.
They are neither intrinsic semantic head types nor SR-safety certificates.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


LABELS = ("low_impact", "action_sensitive", "video_sensitive", "mixed_sensitive", "intermediate")


def assign(action, video, thresholds):
    ah = action > thresholds["action_q75"]
    vh = video > thresholds["video_q75"]
    low = action <= thresholds["action_q25"] and video <= thresholds["video_q25"]
    if ah and vh:
        return "mixed_sensitive"
    if ah:
        return "action_sensitive"
    if vh:
        return "video_sensitive"
    return "low_impact" if low else "intermediate"


def rank_correlation(left, right):
    def ranks(values):
        ordered = np.sort(values)
        return (np.searchsorted(ordered, values, side="left") +
                np.searchsorted(ordered, values, side="right") - 1) / 2
    a, b = ranks(left), ranks(right)
    return float(np.corrcoef(a,b)[0,1]) if np.std(a) > 0 and np.std(b) > 0 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out-dir", type=Path, help="separate analysis output; raw collection is never changed")
    parser.add_argument("--attention-dir", type=Path, help="optional Dense AV/VV statistics on the identical frozen inputs")
    args = parser.parse_args()
    root=args.run_dir
    out=args.out_dir or root
    manifest=json.loads((root/"manifest.json").read_text())
    if manifest["status"] != "MEASURED" or manifest["completed_interventions"] != manifest["planned_interventions"]:
        raise ValueError("refuse to classify an incomplete sweep")
    inputs=[entry["id"] for entry in manifest["inputs"]]
    if len(inputs) != 3:
        raise ValueError("this prespecified first analysis expects two calibration inputs and one check input")
    records=[json.loads(line) for line in (root/"interventions.jsonl").read_text().splitlines()]
    index={(r["input_id"],r["layer"],r["head"],r["stage"]):r for r in records}
    units=[(l,h,s) for l in manifest["layers"] for h in manifest["heads"] for s in range(3)]
    if len(index) != len(records) or len(index) != 3*len(units):
        raise ValueError("duplicate or missing intervention identities")
    arrays=np.asarray([[[index[(i,*u)]["normalized_action_relative_l2"],
                          index[(i,*u)]["future_video_relative_l2"]] for u in units] for i in inputs])
    if not np.isfinite(arrays).all():
        raise ValueError("non-finite sensitivity")
    calibration=arrays[:2].mean(axis=0)
    q25,q75=np.quantile(calibration,[0.25,0.75],axis=0)
    thresholds=dict(action_q25=float(q25[0]), action_q75=float(q75[0]),
                    video_q25=float(q25[1]), video_q75=float(q75[1]))
    labels=[assign(a,v,thresholds) for a,v in calibration]
    checked=[assign(a,v,thresholds) for a,v in arrays[2]]
    entries=[]
    for j,(layer,head,stage) in enumerate(units):
        entries.append(dict(layer=layer,head=head,stage=stage,label=labels[j],
            calibration_action_relative_l2=float(calibration[j,0]),
            calibration_video_relative_l2=float(calibration[j,1]),
            check_action_relative_l2=float(arrays[2,j,0]),check_video_relative_l2=float(arrays[2,j,1]),
            check_label=checked[j],label_agrees=labels[j]==checked[j]))
    confusion={a:{b:sum(x==a and y==b for x,y in zip(labels,checked)) for b in LABELS} for a in LABELS}
    per_stage=[]
    for stage in range(3):
        positions=[j for j,u in enumerate(units) if u[2]==stage]
        left,right=calibration[positions],arrays[2,positions]
        per_stage.append(dict(stage=stage,step_range=manifest["stages"][stage],
            units=len(positions),counts=dict(Counter(labels[j] for j in positions)),
            check_counts=dict(Counter(checked[j] for j in positions)),
            check_label_agreement=float(np.mean([labels[j]==checked[j] for j in positions])),
            action_rank_correlation=rank_correlation(left[:,0],right[:,0]),
            video_rank_correlation=rank_correlation(left[:,1],right[:,1]),
            action_video_rank_correlation=rank_correlation(left[:,0],left[:,1]),
            calibration_action_median=float(np.median(left[:,0])),
            calibration_video_median=float(np.median(left[:,1]))))
    report=dict(status="EXPLORATORY_CLASSIFICATION",utc=datetime.now(timezone.utc).isoformat(),
        argv=sys.argv,analysis_git=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        intervention_sha256=hashlib.sha256((root/"interventions.jsonl").read_bytes()).hexdigest(),
        collector_git=manifest["git"],input_manifest_sha256=manifest["input_manifest_sha256"],
        checkpoint_sha256=manifest["checkpoint_sha256"],future_keep_ratio=manifest["future_keep_ratio"],
        calibration_inputs=inputs[:2],check_input=inputs[2],units=len(units),
        rule="mean of two inputs; high strictly above empirical 75th percentile, low at/below both 25th percentiles; no SR threshold",
        thresholds=thresholds,counts=dict(Counter(labels)),check_counts=dict(Counter(checked)),
        check_label_agreement=float(np.mean(np.asarray(labels)==np.asarray(checked))),
        action_rank_correlation=rank_correlation(calibration[:,0],arrays[2,:,0]),
        video_rank_correlation=rank_correlation(calibration[:,1],arrays[2,:,1]),
        between_calibration_action_rank_correlation=rank_correlation(arrays[0,:,0],arrays[1,:,0]),
        action_video_rank_correlation=rank_correlation(calibration[:,0],calibration[:,1]),
        per_stage=per_stage,
        confusion=confusion,
        limitations=["Three exposed official initial observations; no held-out SR claim",
                     "One strong future-VV restriction, not a calibrated deployable budget",
                     "Stages contain 3, 4 and 3 interventions; raw stage differences also reflect this dose",
                     "Relative quartile labels do not establish natural clusters or intrinsic head semantics",
                     "Combined interventions and paired SR remain required before enabling a policy"],
        sr=None,entries=entries)
    if args.attention_dir:
        attn_manifest=json.loads((args.attention_dir/"manifest.json").read_text())
        if attn_manifest["status"] != "MEASURED" or any(attn_manifest[key] != manifest[key]
                for key in ["checkpoint_sha256","input_manifest_sha256"]):
            raise ValueError("attention statistics do not match the completed sensitivity inputs/checkpoint")
        source=args.attention_dir/"head-stage-attention.json"
        attention_sha=hashlib.sha256(source.read_bytes()).hexdigest()
        if attention_sha != attn_manifest["artifacts"][source.name]:
            raise ValueError("attention artifact changed")
        attention=json.loads(source.read_text())
        lookup={(r["input_id"],r["layer"],r["head"],r["stage"]):r for r in attention}
        if len(lookup) != len(attention) or set(lookup) != set(index):
            raise ValueError("attention statistics have missing, extra or duplicate identities")
        metrics=[k for k in attention[0] if k not in {"input_id","layer","head","stage"}]
        correlations={}
        for metric in metrics:
            values=np.asarray([[lookup[(i,*u)][metric] for u in units] for i in inputs])
            if not np.isfinite(values).all():
                raise ValueError("non-finite attention statistics")
            train=values[:2].mean(axis=0)
            for j,entry in enumerate(entries):
                entry["calibration_"+metric]=float(train[j])
                entry["check_"+metric]=float(values[2,j])
            correlations[metric]=dict(
                calibration_action_rank_correlation=rank_correlation(train,calibration[:,0]),
                calibration_video_rank_correlation=rank_correlation(train,calibration[:,1]),
                cross_input_rank_correlation=rank_correlation(train,values[2]))
        report["attention_statistics"]=dict(collector_git=attn_manifest["git"],
            sha256=attention_sha,definitions=attn_manifest["definitions"],correlations=correlations,
            interpretation="descriptive correlations; class labels remain solely the prespecified perturbation sensitivities")
    if args.out_dir:out.mkdir(parents=True,exist_ok=False)
    (out/"classification.json").write_text(json.dumps(report,indent=2)+"\n")
    with (out/"head-stage.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(entries[0]));writer.writeheader();writer.writerows(entries)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    layer_index={value:i for i,value in enumerate(manifest["layers"])}
    head_index={value:i for i,value in enumerate(manifest["heads"])}
    shape=(len(layer_index),len(head_index))
    fig,axes=plt.subplots(3,3,figsize=(15,15),constrained_layout=True)
    floors=[]
    limits=[]
    for metric in range(2):
        values=calibration[:,metric]
        positive=values[values>0]
        floor=max(float(positive.min())/10,1e-12) if len(positive) else 1e-12
        shown=np.log10(np.maximum(values,floor))
        floors.append(floor)
        limits.append((float(shown.min()),float(shown.max())))
    for stage in range(3):
        for metric in range(2):
            grid=np.full(shape,np.nan)
            for j,(layer,head,s) in enumerate(units):
                if s==stage:grid[layer_index[layer],head_index[head]]=calibration[j,metric]
            shown=np.log10(np.maximum(grid,floors[metric]))
            im=axes[stage,metric].imshow(shown,aspect="auto",cmap="viridis",
                vmin=limits[metric][0],vmax=limits[metric][1])
            lo,hi=manifest["stages"][stage]
            axes[stage,metric].set_title(f"Stage {stage} (steps {lo}–{hi-1}): log10 {'action' if metric==0 else 'video'} L2")
            fig.colorbar(im,ax=axes[stage,metric],shrink=0.8)
        grid=np.zeros(shape)
        for j,(layer,head,s) in enumerate(units):
            if s==stage:grid[layer_index[layer],head_index[head]]=LABELS.index(labels[j])
        im=axes[stage,2].imshow(grid,aspect="auto",vmin=-0.5,vmax=4.5,
            cmap=ListedColormap(["#b9c5cd","#cc503e","#4477aa","#aa4499","#f0c76b"]))
        axes[stage,2].set_title(f"Stage {stage}: relative class")
        bar=fig.colorbar(im,ax=axes[stage,2],ticks=range(5),shrink=0.8)
        bar.ax.set_yticklabels(LABELS,fontsize=8)
        for axis in axes[stage]:
            axis.set_xlabel("Head index");axis.set_ylabel("Layer index")
            for values,setter in [(manifest["heads"],axis.set_xticks),(manifest["layers"],axis.set_yticks)]:
                positions=list(range(0,len(values),max(1,len(values)//6)))
                if positions[-1] != len(values)-1:positions.append(len(values)-1)
                setter(positions,[str(values[i]) for i in positions])
    fig.suptitle("DreamWAM M1 offline sensitivity — exploratory, no SR guarantee",fontsize=15)
    fig.savefig(out/"head-stage-sensitivity.png",dpi=180)
    fig.savefig(out/"head-stage-sensitivity.pdf")
    plt.close(fig)
    print(json.dumps({k:v for k,v in report.items() if k!="entries"},indent=2))


if __name__ == "__main__":
    main()
