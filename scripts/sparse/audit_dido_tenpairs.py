#!/usr/bin/env python3
"""Read-only aggregate of two disjoint, complete matched continuation lanes."""
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
    parser.add_argument('--initial-states', type=int, nargs='+', default=[3])
    parser.add_argument('--expected-cap', type=int, default=50)
    args = parser.parse_args()
    if len(set(args.initial_states)) != len(args.initial_states) or min(args.initial_states) < 0:
        parser.error('initial states must be distinct nonnegative indices')
    n = 10 * len(args.initial_states)
    args.out.mkdir(parents=True, exist_ok=False)
    start = datetime.now(timezone.utc).isoformat()
    arms = {arm: dict(rows={}, warm=[], all=[], balanced=[], env=[], wall=[]) for arm in ('dense', 'sparse')}
    lanes, inputs, provenance = {}, [], set()
    protocol = None
    for lane in ('a', 'b'):
        root = args.root / f'lane-{lane}'
        verified = audit(root, expected_cap=args.expected_cap)
        (args.out / f'lane-{lane}-integrity.json').write_text(json.dumps(verified, indent=2) + '\n')
        provenance.add((verified['source_model_commit'], verified['evaluator_commit'], verified['checkpoint_sha256']))
        controller = json.loads((root / 'controller.json').read_text())
        assert controller['render_backend'] == 'osmesa'
        lane_inputs, lane_times = {}, {}
        for arm in arms:
            run = root / arm / 'run'
            manifest, planned = load_manifest(run)
            assert planned == {(i, init, 0, 42) for i in (range(5) if lane == 'a' else range(5, 10)) for init in args.initial_states}
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
    assert all(row['state_equal'] and row['agentview_equal'] and row['wrist_equal'] for row in inputs)
    expected = {(i, init, 0, 42) for i in range(10) for init in args.initial_states}
    assert all(set(a['rows']) == expected for a in arms.values())
    per_task, outcomes = {}, []
    for identity in sorted(expected):
        d, s = (arms[arm]['rows'][identity] for arm in ('dense', 'sparse'))
        per_task.setdefault(identity[0], []).append((d['success'], s['success']))
        outcomes.append(dict(identity=identity, dense=d, sparse=s))
    successes = {arm: sum(row['success'] for row in data['rows'].values()) for arm, data in arms.items()}
    b = sum(d and not s for pairs in per_task.values() for d, s in pairs)
    c = sum(s and not d for pairs in per_task.values() for d, s in pairs)
    two, one = mcnemar_exact(b, c)
    timings = {arm: dict(warm=distribution(data['warm']), all_calls=distribution(data['all']),
        episode_balanced_warm_mean=statistics.fmean(data['balanced']),
        environment=distribution(data['env']), wall=distribution(data['wall'])) for arm, data in arms.items()}
    eligible = all(len(pairs) >= 2 for pairs in per_task.values())
    bootstrap = (stratified_bootstrap(per_task, resamples=10000, seed=42) if eligible else
        dict(delta_ci95=None, dense_sr_ci95=None, sparse_sr_ci95=None, resamples=0))
    bootstrap['bootstrap_withheld_reason'] = None if eligible else 'one episode per task; within-task resampling would be degenerate'
    interval = bootstrap['delta_ci95']
    bootstrap['bootstrap_degenerate'] = interval is not None and interval[0] == interval[1]
    result = dict(status=f'VERIFIED_COMPLETE_{n}_PAIRS', start_utc=start,
        end_utc=datetime.now(timezone.utc).isoformat(), exit_code=0,
        auditor_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        source_model_commit=next(iter(provenance))[0], evaluator_commit=next(iter(provenance))[1],
        checkpoint_sha256=next(iter(provenance))[2], protocol=protocol, new_attempts=2*n,
        success_counts=successes, success_rate={arm: count / n for arm, count in successes.items()},
        wilson95={arm: wilson(count, n) for arm, count in successes.items()},
        delta_percentage_points=100 * (successes['sparse'] - successes['dense']) / n,
        sr_margin_established=False, paired=dict(dense_only=b, sparse_only=c, unchanged=n-b-c,
            mcnemar_two_sided_p=two, mcnemar_sparse_worse_p=one,
            **bootstrap),
        per_task={task: dict(pairs=len(pairs), dense_success=sum(d for d,s in pairs), sparse_success=sum(s for d,s in pairs)) for task,pairs in per_task.items()},
        outcomes=outcomes, initial_inputs=inputs, timings=timings, lanes=lanes,
        descriptive_warm_model_speedup=timings['dense']['warm']['mean'] / timings['sparse']['warm']['mean'],
        descriptive_episode_balanced_warm_speedup=timings['dense']['episode_balanced_warm_mean'] / timings['sparse']['episode_balanced_warm_mean'],
        limitations=['Exploratory fixed cohort; observed SR and intervals do not alone certify the five-point population margin.',
            'Closed-loop observations and lengths may differ; model latency ratio is descriptive, not same-input replay.',
            'Model timing excludes model loading, IPC, physics and CPU rendering.',
            'H100 GPUs3/4 with CPU OSMesa; no pooling with H200 or historical cohorts.'])
    result['integrity_artifacts'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.out.glob('*-integrity.json')}
    (args.out / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: result[key] for key in ('status', 'new_attempts', 'success_counts', 'delta_percentage_points', 'descriptive_warm_model_speedup', 'sr_margin_established')}))


if __name__ == '__main__':
    main()
