#!/usr/bin/env python3
"""Read-only aggregate of two disjoint, complete five-pair continuation lanes."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import subprocess

from audit_dido_pilot import audit
from audit_initial_inputs import read_inputs
from paired_sr import load_episodes, load_manifest, mcnemar_exact, stratified_bootstrap, wilson
from summarize_policy_timings import distribution, inspect_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    start = datetime.now(timezone.utc).isoformat()
    arms = {arm: dict(rows={}, warm=[], all=[], balanced=[], env=[], wall=[]) for arm in ('dense', 'sparse')}
    lanes, inputs, provenance = {}, [], set()
    protocol = None
    for lane in ('a', 'b'):
        root = args.root / f'lane-{lane}'
        verified = audit(root)
        (args.out / f'lane-{lane}-integrity.json').write_text(json.dumps(verified, indent=2) + '\n')
        provenance.add((verified['source_model_commit'], verified['evaluator_commit'], verified['checkpoint_sha256']))
        controller = json.loads((root / 'controller.json').read_text())
        assert controller['render_backend'] == 'osmesa'
        lane_inputs, lane_times = {}, {}
        for arm in arms:
            run = root / arm / 'run'
            manifest, planned = load_manifest(run)
            assert planned == {(i, 3, 0, 42) for i in (range(5) if lane == 'a' else range(5, 10))}
            protocol = manifest['protocol'] if protocol is None else protocol
            assert manifest['protocol'] == protocol
            rows = load_episodes(run)
            assert set(rows) == planned and not (set(rows) & set(arms[arm]['rows']))
            assert all(row['status'] in ('succeeded', 'failed') for row in rows.values())
            arms[arm]['rows'].update(rows)
            _, _, lane_inputs[arm] = read_inputs(run)
            assert set(lane_inputs[arm]) == planned
            _, _, episodes, lane_times[arm] = inspect_run(run)
            assert lane_times[arm]['coverage']['complete']
            arms[arm]['balanced'].extend(row['warm_model']['mean'] for row in episodes.values())
            for path in sorted(run.glob('episodes/*/attempts/*/result.json')):
                result = json.loads(path.read_text())
                calls = json.loads((run / result['artifacts']['policy_calls']).read_text())
                arms[arm]['env'].append(result['env_seconds'])
                arms[arm]['wall'].append(result['wall_seconds'])
                for call in calls:
                    seconds = call['metadata']['timing']['predict_seconds']
                    arms[arm]['all'].append(seconds)
                    if call['call_index'] > 1 and call['metadata']['diagnostics']['prompt_cache']['last_hit']:
                        arms[arm]['warm'].append(seconds)
        for identity in sorted(planned):
            left, right = (lane_inputs[arm][identity]['first'] for arm in ('dense', 'sparse'))
            inputs.append(dict(identity=identity, state_equal=left['state'] == right['state'],
                agentview_equal=left['images']['agentview'] == right['images']['agentview'],
                wrist_equal=left['images']['wrist'] == right['images']['wrist']))
        lanes[lane] = dict(timings=lane_times,
            descriptive_warm_speedup=lane_times['dense']['model_seconds_warm']['mean'] / lane_times['sparse']['model_seconds_warm']['mean'])
    assert len(provenance) == 1
    expected = {(i, 3, 0, 42) for i in range(10)}
    assert all(set(a['rows']) == expected for a in arms.values())
    per_task, outcomes = {}, []
    for identity in sorted(expected):
        d, s = (arms[arm]['rows'][identity] for arm in ('dense', 'sparse'))
        per_task[identity[0]] = [(d['success'], s['success'])]
        outcomes.append(dict(identity=identity, dense=d, sparse=s))
    successes = {arm: sum(row['success'] for row in data['rows'].values()) for arm, data in arms.items()}
    b = sum(d and not s for pairs in per_task.values() for d, s in pairs)
    c = sum(s and not d for pairs in per_task.values() for d, s in pairs)
    two, one = mcnemar_exact(b, c)
    timings = {arm: dict(warm=distribution(data['warm']), all_calls=distribution(data['all']),
        episode_balanced_warm_mean=statistics.fmean(data['balanced']),
        environment=distribution(data['env']), wall=distribution(data['wall'])) for arm, data in arms.items()}
    result = dict(status='VERIFIED_COMPLETE_10_PAIRS', start_utc=start,
        end_utc=datetime.now(timezone.utc).isoformat(), exit_code=0,
        auditor_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        source_model_commit=next(iter(provenance))[0], evaluator_commit=next(iter(provenance))[1],
        checkpoint_sha256=next(iter(provenance))[2], protocol=protocol, new_attempts=20,
        success_counts=successes, success_rate={arm: count / 10 for arm, count in successes.items()},
        wilson95={arm: wilson(count, 10) for arm, count in successes.items()},
        delta_percentage_points=10 * (successes['sparse'] - successes['dense']),
        sr_margin_established=False, paired=dict(dense_only=b, sparse_only=c, unchanged=10-b-c,
            mcnemar_two_sided_p=two, mcnemar_sparse_worse_p=one,
            **stratified_bootstrap(per_task, resamples=10000, seed=42)),
        outcomes=outcomes, initial_inputs=inputs, timings=timings, lanes=lanes,
        descriptive_warm_model_speedup=timings['dense']['warm']['mean'] / timings['sparse']['warm']['mean'],
        descriptive_episode_balanced_warm_speedup=timings['dense']['episode_balanced_warm_mean'] / timings['sparse']['episode_balanced_warm_mean'],
        limitations=['One episode per task; ten pairs cannot certify a five-percentage-point SR margin.',
            'Closed-loop observations and lengths may differ; model latency ratio is descriptive, not same-input replay.',
            'Model timing excludes model loading, IPC, physics and CPU rendering.',
            'H100 GPUs3/4 with CPU OSMesa; no pooling with H200 or historical cohorts.'])
    result['integrity_artifacts'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.out.glob('*-integrity.json')}
    (args.out / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: result[key] for key in ('status', 'new_attempts', 'success_counts', 'delta_percentage_points', 'descriptive_warm_model_speedup', 'sr_margin_established')}))


if __name__ == '__main__':
    main()
