#!/usr/bin/env python3
"""Compare combined M1 allocations at exactly equal per-layer/stage VV budgets.

This is an offline action/video proxy experiment, not an optimized sparse
backend or an SR test. Allocation uses only the two calibration inputs.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import torch

from benchmark_ffn_context_cache import command, sha256
from collect_action_impact import components
from collect_head_stage_impact import relative_l2
from dreamwam.config import load_release_config
from dreamwam.policy import build_policy
from dreamwam.sparse.head_stage_probe import HeadStageProbe, stage_bounds


def allocated_mask(native, budgets, *, video_length, tokens_per_frame, block_size=14):
    """Keep native action/conditioning rows and uniformly spaced future blocks."""
    if native.ndim != 2 or native.dtype != torch.bool or native.shape[0] != native.shape[1]:
        raise ValueError("expected square native boolean mask")
    future=video_length-tokens_per_frame
    if future <= 0 or future % block_size:
        raise ValueError("this equal-cost experiment requires complete future blocks")
    blocks=future//block_size
    if not budgets or any(isinstance(k,bool) or not isinstance(k,int) or not 0 <= k <= blocks for k in budgets):
        raise ValueError("invalid integer per-head future-block budgets")
    mask=native[None,None].expand(1,len(budgets),-1,-1).clone()
    key_blocks=torch.arange(future,device=native.device)//block_size
    for head,keep in enumerate(budgets):
        selected=(torch.linspace(0,blocks-1,keep,device=native.device).round().long()
                  if keep else torch.empty(0,dtype=torch.long,device=native.device))
        membership=(key_blocks[:,None]==selected[None]).any(dim=1)
        mask[0,head,tokens_per_frame:video_length,tokens_per_frame:video_length] &= membership[None]
    return mask


class AllocationController:
    """Installed beneath the existing probe, which captures final outputs."""
    def __init__(self,mot):
        self.mot=mot
        self.profile=None
        self.masks={}
        self.calls=0

    def configure(self,profile):
        self.profile=profile
        self.masks.clear()
        self.calls=0

    def __enter__(self):
        self.original=self.mot._joint_self_attention

        def attention(**kwargs):
            self.calls+=1
            if self.profile is None:return self.original(**kwargs)
            mask=kwargs["attention_mask"]
            if mask.ndim != 4 or mask.shape[:2] != (1,self.mot.num_heads):
                raise ValueError("run allocation underneath the full-budget probe")
            stage=next(s for s,(lo,hi) in enumerate(stage_bounds(kwargs["num_steps"]))
                       if lo <= kwargs["step_index"] < hi)
            budgets=tuple(self.profile[kwargs["layer_index"]][stage])
            if budgets not in self.masks:
                self.masks[budgets]=allocated_mask(mask[0,0],list(budgets),
                    video_length=kwargs["video_length"],tokens_per_frame=kwargs["tokens_per_frame"])
            return self.original(**{**kwargs,"attention_mask":self.masks[budgets]})

        self.mot._joint_self_attention=attention
        return self

    def __exit__(self,*args):
        self.mot._joint_self_attention=self.original
        self.masks.clear()


def allocations(entries,layers=30,heads=24):
    if heads != 24:raise ValueError("the fixed allocation comparison uses 24 heads")
    index={(r['layer'],r['head'],r['stage']):r for r in entries}
    identities={(l,h,s) for l in range(layers) for h in range(heads) for s in range(3)}
    if len(index) != len(entries) or set(index) != identities:
        raise ValueError("classification does not cover all model units exactly once")
    scores=np.asarray([[[index[(l,h,s)]['calibration_action_relative_l2']
                        for h in range(heads)] for s in range(3)] for l in range(layers)])
    if not np.isfinite(scores).all():raise ValueError("non-finite calibration sensitivity")
    dense=np.full((layers,3,heads),14,dtype=int)
    uniform=np.full_like(dense,7)
    head_only=np.zeros_like(dense)
    head_stage=np.zeros_like(dense)
    top=lambda score:np.lexsort((np.arange(heads),-score))[:heads//2]
    for layer in range(layers):
        head_only[layer,:,top(scores[layer].mean(axis=0))]=14
        for stage in range(3):head_stage[layer,stage,top(scores[layer,stage])]=14
    profiles=dict(dense=dense,uniform=uniform,head_only=head_only,head_stage=head_stage)
    for seed in (0,1,2):
        rng=np.random.default_rng(seed)
        random=np.zeros_like(dense)
        for layer in range(layers):
            for stage in range(3):random[layer,stage,rng.choice(heads,heads//2,replace=False)]=14
        profiles[f'random_{seed}']=random
    if any(not np.all(p.sum(axis=-1)==168) for name,p in profiles.items() if name!='dense'):
        raise AssertionError("per-layer/stage costs are not identical")
    return {k:v.tolist() for k,v in profiles.items()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/dreamwam_joint.yaml')
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--classification',type=Path,required=True)
    parser.add_argument('--out-dir',type=Path,required=True)
    args=parser.parse_args()
    dataset=json.loads(args.inputs.read_text())
    classes=json.loads(args.classification.read_text())
    ids=[r['id'] for r in dataset['inputs']]
    if (dataset['status']!='CAPTURED' or len(ids)!=3 or classes['status']!='EXPLORATORY_CLASSIFICATION'
        or classes['calibration_inputs']!=ids[:2] or classes['check_input']!=ids[2]
        or classes['input_manifest_sha256']!=sha256(args.inputs)):
        raise ValueError('require complete classification on these two calibration inputs and one check input')
    release=load_release_config(args.config)
    if classes['checkpoint_sha256']!=sha256(release.paths.checkpoint):raise ValueError('checkpoint mismatch')
    profiles=allocations(classes['entries'])
    args.out_dir.mkdir(parents=True,exist_ok=False)
    report=dict(status='RUNNING',start_utc=datetime.now(timezone.utc).isoformat(),argv=sys.argv,
        pid=os.getpid(),git=command('git','rev-parse','HEAD'),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        checkpoint_sha256=classes['checkpoint_sha256'],input_manifest_sha256=sha256(args.inputs),
        classification_sha256=sha256(args.classification),calibration_inputs=ids[:2],check_input=ids[2],
        sources={p:sha256(p) for p in [__file__,'dreamwam/sparse/head_stage_probe.py','dreamwam/mot.py',args.config]},
        profiles=profiles,records=[],sr=None,
        rule='protect the 12 most action-sensitive heads per layer; head-only averages three stages, head-stage ranks each stage separately; ascending head index breaks ties',
        cost='all sparse profiles retain 168 future blocks per layer/stage: 50% future keys and 5/7 total VV pairs; no actual compute saving claimed',
        limitations=['One predeclared 50% future-key budget','Only three exposed initial observations',
            'Masked fused Dense execution is diagnostic, not an optimized sparse backend',
            'Proxy differences and deterministic replay do not establish paired SR'])
    write=lambda:(args.out_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    write()
    try:
        policy=build_policy(release,device='cuda')
        versions={k:p._version for k,p in policy.model.named_parameters()}
        evaluation=dict(policy.evaluation)
        if evaluation['denoising_steps']!=10 or evaluation['action_horizon']!=32:raise ValueError('retain sampler protocol')
        for entry in dataset['inputs']:
            source=args.inputs.parent/entry['path']
            if sha256(source)!=entry['sha256']:raise ValueError('input changed')
            with np.load(source,allow_pickle=False) as data:
                images={k:data[k].copy() for k in ['agentview','wrist']}
                state,instruction=data['state'].copy(),str(data['instruction'].item())
            predict=lambda:policy.predict_action(images=images,state=state,instruction=instruction)
            native=predict()
            with AllocationController(policy.model.mot) as controller,HeadStageProbe(policy.model) as probe:
                baseline=predict()
                if not np.array_equal(baseline,native):raise AssertionError('native baseline parity failed')
                video,raw=probe.last_video.clone(),probe.last_raw_action.clone()
                for name,profile in profiles.items():
                    controller.configure(profile)
                    action=predict()
                    actual_video,actual_raw=probe.last_video.clone(),probe.last_raw_action.clone()
                    if not np.isfinite(action).all() or not torch.isfinite(actual_video).all():raise FloatingPointError('non-finite combined output')
                    if name=='dense' and (not np.array_equal(action,native) or not torch.equal(actual_video,video)):
                        raise AssertionError('full allocation does not equal Dense')
                    if controller.calls!=300 or probe.stats['attention_calls']!=300 or probe.stats['video_scheduler_steps']!=10:
                        raise AssertionError('incomplete combined sampling scope')
                    if not np.array_equal(predict(),action) or not torch.equal(probe.last_video,actual_video):
                        raise AssertionError('combined allocation is not repeatable')
                    row=dict(input_id=entry['id'],profile=name,actions=action.tolist(),
                        normalized_action_relative_l2=relative_l2(actual_raw,raw),
                        future_video_relative_l2=relative_l2(actual_video[:,:,1:],video[:,:,1:]),
                        conditioned_frame_bitwise=bool(torch.equal(actual_video[:,:,:1],video[:,:,:1])),
                        action=components(baseline,action),repeat_bitwise=True)
                    if not row['conditioned_frame_bitwise']:raise AssertionError('conditioned frame changed')
                    report['records'].append(row);write()
                controller.configure(None)
                if not np.array_equal(predict(),native) or not torch.equal(probe.last_video,video):raise AssertionError('Dense drifted')
        if evaluation!=policy.evaluation or versions!={k:p._version for k,p in policy.model.named_parameters()}:raise AssertionError('weights/settings changed')
        report.update(status='MEASURED',end_utc=datetime.now(timezone.utc).isoformat(),evaluation=evaluation)
        write()
        print(json.dumps({k:v for k,v in report.items() if k not in ['records','profiles']},indent=2))
    except BaseException as exc:
        report.update(status='ERROR',error=repr(exc),end_utc=datetime.now(timezone.utc).isoformat());write();raise


if __name__=='__main__':main()
