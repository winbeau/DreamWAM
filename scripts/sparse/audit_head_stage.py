#!/usr/bin/env python3
"""Audit a complete M1 record matrix and its published relative classification."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir',type=Path)
    parser.add_argument('analysis_dir',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    manifest=json.loads((args.run_dir/'manifest.json').read_text())
    report=json.loads((args.analysis_dir/'classification.json').read_text())
    if manifest['status']!='MEASURED' or manifest['completed_interventions']!=6480 or manifest['planned_interventions']!=6480:
        raise ValueError('require the complete 6480-intervention experiment')
    inputs=[r['id'] for r in manifest['inputs']]
    expected={(i,l,h,s) for i in inputs for l in range(30) for h in range(24) for s in range(3)}
    records=[json.loads(line) for line in (args.run_dir/'interventions.jsonl').read_text().splitlines()]
    index={(r['input_id'],r['layer'],r['head'],r['stage']):r for r in records}
    assert len(inputs)==3 and len(index)==len(records)==6480 and set(index)==expected
    assert report['intervention_sha256']==digest(args.run_dir/'interventions.jsonl')
    assert report['checkpoint_sha256']==manifest['checkpoint_sha256']
    assert report['input_manifest_sha256']==manifest['input_manifest_sha256']
    assert report['calibration_inputs']==inputs[:2] and report['check_input']==inputs[2]
    assert len(manifest['controls'])==3
    assert {r['input_id'] for r in manifest['controls']}==set(inputs)
    for row in manifest['controls']:
        assert row['native_repeat_bitwise'] and row['full_budget_native_bitwise']
        assert row['reference_stats']['attention_calls']==300 and row['reference_stats']['video_scheduler_steps']==10
    max_metric_recompute_error=0.0
    for input_id in inputs:
        with np.load(args.run_dir/f'reference-{input_id}.npz',allow_pickle=False) as data:
            reference=data['actions'].copy()
            assert reference.shape==(32,7) and np.isfinite(reference).all()
            assert np.isfinite(data['normalized_actions']).all() and np.isfinite(data['video_latents']).all()
        for row in [r for r in records if r['input_id']==input_id]:
            bounds=((0,3),(3,7),(7,10))[row['stage']]
            assert row['step_range']==list(bounds)
            stats=row['stats']
            assert stats['attention_calls']==300 and stats['video_scheduler_steps']==10
            assert stats['target_steps']==list(range(*bounds)) and stats['target_calls']==bounds[1]-bounds[0]
            assert stats['mask']['future_blocks']==14 and stats['mask']['kept_future_blocks']==0
            assert stats['mask']['removed_pairs']==196*196
            assert abs(stats['mask']['target_vv_density']-3/7)<1e-12
            assert row['conditioned_frame_bitwise'] is True
            assert all(np.isfinite(row[k]) and row[k]>=0 for k in ['normalized_action_relative_l2','future_video_relative_l2'])
            actual=np.asarray(row['actions'],dtype=reference.dtype)
            assert actual.shape==(32,7) and np.isfinite(actual).all()
            difference=actual-reference
            recalculated=dict(max_abs=float(np.abs(difference).max()),mean_abs=float(np.abs(difference).mean()),
                relative_l2=float(np.linalg.norm(difference)/max(np.linalg.norm(reference),1e-12)),
                translation_rms=float(np.sqrt(np.mean(difference[:,:3]**2))),
                rotation_rms=float(np.sqrt(np.mean(difference[:,3:6]**2))),
                gripper_sign_flips=int(np.count_nonzero(np.sign(actual[:,-1])!=np.sign(reference[:,-1]))),
                gripper_fraction_unchanged=float(np.mean(np.sign(actual[:,-1])==np.sign(reference[:,-1]))))
            for name,value in recalculated.items():
                assert np.isclose(value,row['action'][name],rtol=1e-6,atol=1e-9),(input_id,name)
                max_metric_recompute_error=max(max_metric_recompute_error,abs(value-row['action'][name]))
    entries=report['entries']
    by_unit={(r['layer'],r['head'],r['stage']):r for r in entries}
    units=sorted({(l,h,s) for _,l,h,s in expected})
    assert len(by_unit)==len(entries)==2160 and set(by_unit)==set(units)
    values=np.asarray([[[index[(i,*u)][k] for k in ['normalized_action_relative_l2','future_video_relative_l2']]
                        for u in units] for i in inputs])
    train=values[:2].mean(axis=0)
    low,high=np.quantile(train,[0.25,0.75],axis=0)
    assert report['thresholds']==dict(action_q25=float(low[0]),action_q75=float(high[0]),video_q25=float(low[1]),video_q75=float(high[1]))
    def classify(pair):
        above=pair>high
        if above.all():return 'mixed_sensitive'
        if above[0]:return 'action_sensitive'
        if above[1]:return 'video_sensitive'
        if (pair<=low).all():return 'low_impact'
        return 'intermediate'
    labels=[classify(x) for x in train]
    checks=[classify(x) for x in values[2]]
    for j,u in enumerate(units):
        entry=by_unit[u]
        assert entry['label']==labels[j] and entry['check_label']==checks[j]
        assert entry['calibration_action_relative_l2']==train[j,0]
        assert entry['calibration_video_relative_l2']==train[j,1]
        assert entry['check_action_relative_l2']==values[2,j,0]
        assert entry['check_video_relative_l2']==values[2,j,1]
    assert report['counts']==dict(Counter(labels)) and report['check_counts']==dict(Counter(checks))
    actual_agreement=sum(x==y for x,y in zip(labels,checks))/len(labels)
    assert abs(report['check_label_agreement']-actual_agreement)<1e-12
    for left,row in report['confusion'].items():
        for right,count in row.items():assert count==sum(a==left and b==right for a,b in zip(labels,checks))
    chance=sum(labels.count(k)*checks.count(k) for k in set(labels)|set(checks))/len(labels)**2
    result=dict(status='VERIFIED',utc=datetime.now(timezone.utc).isoformat(),
        audit_git=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        audit_source_sha256=digest(Path(__file__)),interventions=6480,units=2160,
        input_ids=inputs,manifest_sha256=digest(args.run_dir/'manifest.json'),
        intervention_sha256=digest(args.run_dir/'interventions.jsonl'),classification_sha256=digest(args.analysis_dir/'classification.json'),
        identity_scope_controls_finiteness_and_classification_verified=True,
        denormalized_action_metrics_recomputed=True,max_metric_recompute_error=max_metric_recompute_error,
        check_label_agreement=actual_agreement,marginal_chance_agreement=chance,
        cohen_kappa=(actual_agreement-chance)/(1-chance) if chance<1 else None,
        same_label_by_type={k:dict(total=labels.count(k),stable=sum(a==b==k for a,b in zip(labels,checks))) for k in sorted(set(labels))},
        limitations=['Normalized-action and video scalar errors are audited as recorded outputs of the verified collector; intervention latent tensors were not retained',
            'Classification agreement is on one related exposed initial observation, not held-out SR'],sr=None)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
