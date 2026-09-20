# Nine-observation retained-choice comparison

Status: all 277 predictions and 126 warm timing rows verified on H100,
2026-09-20 20:48:37–20:50:52 UTC, exit 0. Source and auditor:
`41f515a5480cdcc87a18b147ab393e9cae337e9f`. Existing checkpoint SHA-256 remains
`6c087e5b9e201f19dbe92834b470a7ee18937a18ea9adcbb49886aa1e3ac4c61`.
These nine self-captured observations are three positions in each of three
exposed development trajectories. They are not nine independent episodes.

The apparent benefit on two middle-trajectory inputs does **not** generalize to
the full development set. The new native selectors have substantially larger
action error and more gripper differences than the inherited uniform selector,
at similar warm latency. No new selector should replace uniform on this evidence.

| Method | Paired warm speedup vs Dense | vs inherited uniform | Mean raw prefix L2 | Worst | Gripper differences / 90 |
|---|---:|---:|---:|---:|---:|
| Inherited uniform R56 feature reuse | 2.118× across both groups | 1.000× | 0.335966 | 1.102866 | 6 |
| Shared context R56 | 2.094× | 1.000× | 0.711185 | 1.460379 | 35 |
| Shared context R84 | 2.131× | 1.018× | 0.715322 | 1.458795 | 36 |
| Layerwise value-aware R56 | 2.130× | 0.994× | 0.701120 | 1.456085 | 35 |

Each candidate has 18 warm observations (two timing repeats on each of nine
inputs). Mean/p50/p95 complete prediction latency is 160.03/158.96/178.08 ms
for shared R56, 157.28/158.18/186.63 ms for shared R84, and
153.46/155.77/163.97 ms for layerwise R56. Each ratio uses that candidate's
matched group controls; absolute latencies from different groups reflect shared
GPU load and do not imply a new selector is faster than uniform.

All three native methods disagree with Dense on all ten prefix gripper commands
at each trajectory's initial observation. The middle observations are much closer;
late observations include additional failures of the action proxy. These are
action differences, not benchmark task failures. The selected diagnostic pilot
candidate, layerwise R56, is the least-error choice among the **new** candidates;
it is not better than uniform. Its subsequently measured control consequence is
in the [final report](REPORT.md); no retuning followed this frozen screen.

The speed comes from D0/R1–9 feature reuse. Every action token executes all 30
layers at all ten denoising steps; only D0 computes all 294 visual rows. Native
layerwise scores and packing are included online. R1–9 read 56 rows/layer and
perform zero visual queries. The nominal 10% recompute budget is inactive.
The [fresh-structure control](FOLLOWUP-RESULTS.md) did not reach 1.5×.

All graph/setup calls match their own eager raw/executable arrays byte for byte.
The independent audit also checks 45 trace calls / 450 steps. Route examples for
input `t000-i001-r00-s42-c0001`, eager call 157, D0 include:

- Layer 0 observed-frame IDs: `[3,4,6,16,17,18,32,46,47,60,61,62,63,75,76,88,89,95,96]`.
- Layer 29 observed-frame IDs: `[0,11,14,19,20,28,29,32,33,42,43,44,45,46,47,57,59,60,95]`.
- Every layer uses frame quotas `[19,19,18]`. Full routes for all layers and steps
  are archived, with D0 route hash
  `6b91389b28473636f6ac8cb7d692371c0dba7c7110b6ca99008c4cd55d83e3cc`.

For flat index `i`, frame=`i//98`, row=`(i%98)//14`, concatenated column=`i%14`;
columns 0–6 are agentview and 7–13 wrist. Thus ID 95 is frame 0, row 6,
wrist column 4. This is model-grid mapping, not semantic contact annotation.

Model loading takes 14.13 s and peak allocated/reserved memory is
26,473,067,008 / 27,005,026,304 bytes. First graph call (already loaded process)
for layerwise R56 is 0.932 s versus matched Dense 1.070 s and uniform 0.754 s.
Those are not warm times or full process-start latency. Adapter IPC and whole-
episode timing remain separate and cannot inherit the approximately 2× claim.

Artifacts under H100 `outputs/dido-sparse-profile-20260920/`:
`native-final-41f515a/{report.json,calls.jsonl,requests.jsonl,summary.json,actions/}`
and `native-final-audit-41f515a/{report.json,per-input-timings.csv}`. Reproduction:
`python scripts/sparse/benchmark_native_routes.py --stage frozen --design-key
online_final --share-gpu5 --admission-wait-seconds 120 --out-dir <new-directory>`.
This records a completed finite run; it does not authorize duplicate testing.

The completed six-attempt development pilot uses model `41f515a`, evaluator
`0f856c9`, tasks 0/1/2 × initial state 1, unchanged release protocol and CPU
OSMesa: Dense/candidate both 3/3. The final fixed cohort uses evaluator `b5952a5`
and tasks 3/4/5 × initial state 2: Dense 2/3, candidate 3/3. The study stopped
at 12 attempts. The five-point SR tolerance is not established by offline errors
or these tiny cohorts. No training, checkpoint or dependency changed.
