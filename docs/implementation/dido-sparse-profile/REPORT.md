# DIDO-guided DreamWAM study: bounded final report

Status: finite investigation completed on H100, 2026-09-20 UTC. The last rollout
ended at 21:25:01; all audits passed and owned processes were absent at 21:29:01.
The scientific target is **not established**: warm full prediction exceeds 1.5×,
but the new selection rules do not improve the final offline action proxies over
inherited uniform, and twelve real attempts cannot certify the user's five-point
SR margin. No further episodes are scheduled. This is an exploratory study, not
a benchmark or a recommendation to replace the existing uniform candidate.

[FINAL-VERIFICATION.json](FINAL-VERIFICATION.json) contains exact revisions,
timestamps, commands, exit codes, artifact hashes, timing distributions, the
12-slot ledger snapshot and resource-release evidence. Earlier immutable records
remain in [VERIFICATION.json](VERIFICATION.json),
[ONLINE-VERIFICATION.json](ONLINE-VERIFICATION.json) and
[FOLLOWUP-VERIFICATION.json](FOLLOWUP-VERIFICATION.json).

## What was tested

The frozen candidate uses layerwise value-aware selection at D0, then reads
56/294 visual rows per layer during R1–9 with quotas `[19,19,18]`. Its score is
mean action-query attention to a visual key multiplied by that key's V norm,
averaged over four actual native heads at each layer. The original joint
Softmax denominator, masks and positions are retained. This is our proxy,
not the DIDO author's unavailable attribution implementation. All 32 action
tokens execute all 30 layers at all ten denoising steps.

The approximately 2× speed comes from **visual feature reuse**. R1–9 execute
zero visual queries and reuse D0 features/K/V. The nominal 10% recompute setting
is inactive in this schedule. It does not establish equivalent speed with fresh
visual features every step. Selection, indexing, packing, transfers and model
outputs are included in complete prediction timing; offline tensor archival and
analysis are outside it. Both controls use the same checkpoint, backend,
conditioned-frame optimization and exact prompt cache.

The implementation separately exposes action/value/dynamic/context signals,
read and recompute budgets, feature versus structure reuse, explicit refresh
steps, frame treatment and background pooling. Shared hard routes enforce
`R_new - R_old ⊆ U ⊆ R_new`. Layerwise and pooled routes support D/R feature
schedules; unsupported Sparse/structure combinations fail explicitly. Pooling
preserves camera/frame boundaries and mask visibility, averages post-RoPE K and
unrotated V, and optionally adds log multiplicity. It is a separately measured
approximation, not hard deletion. See [PROFILE.md](PROFILE.md) for formulas and
[ONLINE.md](ONLINE.md) for implementation and numerical semantics.

## Data and diagnostic findings

No author demonstrations, boxes/tracks, semantic contact labels or attribution
raw records were obtained; the user confirmed there is no private link. The
[versioned source audit](SOURCES.md) distinguishes the paper, unpublished code,
training-dependent methods and inference-time refinement. This work performs no
training, distillation, checkpoint change or DIDO-model reproduction.

Raw data are our nine DreamWAM observations from three exposed development
trajectories, plus native Q/K/V, masks, action/video outputs and runtime traces.
They are correlated samples, not nine independent episodes. Capture covers all
ten steps, sampled depths `[0,9,19,29]` and heads `[0,6,12,18]`, with bitwise
native-output parity. All 567 NPZ files (785,128,759 compressed bytes) and
423,360 exported score rows pass independent replay. No future ground truth,
semantic labels or simulator privileged state enters online selection.

Action-read support is relatively stable between neighboring denoising steps
(mean Jaccard 0.896), but varies strongly across sampled depths (0.150) and heads
(0.175). Value dynamics and attention are distinct proxies. Forty-seven bounded
AV/VV/joint interventions on two middle observations distinguish direct action
reads from visual context; they show no consistently superior selector. They
cover a partial design, with jointly varied early/shallow and late/deep endpoints;
they cannot independently establish a layer effect, causal contact importance or
SR. [RESULTS.md](RESULTS.md) contains every diagnostic and its limitations.

## Same-input performance and unfavorable results

The final screen uses all nine development observations with two timing repeats
per input and candidate, interleaved with matched Dense and inherited uniform.
All 277 complete calls, 126 warm timing rows, 45 route traces/450 steps and raw
eager/graph arrays pass independent replay. The apparent improvement on two
middle inputs reverses across the full set:

| Method | Warm speedup vs matched Dense | Mean / worst raw-prefix relative L2 | Prefix gripper differences |
|---|---:|---:|---:|
| Inherited uniform R56 | 2.118× across both groups | 0.335966 / 1.102866 | 6/90 |
| Shared context R56 | 2.094× | 0.711185 / 1.460379 | 35/90 |
| Shared context R84 | 2.131× | 0.715322 / 1.458795 | 36/90 |
| Layerwise value-aware R56 | 2.130× | 0.701120 / 1.456085 | 35/90 |

The last row is the least-error **new** candidate, frozen for diagnostic closed
loop evaluation. It is 0.994× as fast as its matched uniform control and has
substantially worse action proxies. Candidate mean/p50/p95 warm latency is
153.46/155.77/163.97 ms. Initial inputs produce ten gripper disagreements each;
these unfavorable cases are retained. Action error does not itself determine SR.

The final screen loads the model in 14.13 s, with peak allocated/reserved memory
26,473,067,008 / 27,005,026,304 bytes. First graph prediction in the already
loaded process is 0.932 s for the candidate, 1.070 s for its Dense control and
0.754 s for uniform. These are separate from warm timing and process startup.
[FINAL-OFFLINE-RESULTS.md](FINAL-OFFLINE-RESULTS.md) includes actual route IDs,
grid mapping, distributions, exact call budgets and reproduction commands.

The earlier [20-candidate screen](ONLINE-RESULTS.md) and
[refresh/budget follow-up](FOLLOWUP-RESULTS.md) remain part of the result:

- Every single Sparse refresh position 1–9 and both measured-rank multi-refresh
  controls worsen mean prefix error against no refresh on the two exposed inputs.
  Read budgets 28/56/84 and recompute budgets 15/30/45 are tested independently.
  Smaller reads bring little additional speed; no adaptive threshold is adopted.
- Fresh retained-Q/K/V structure reuse reaches only 1.255× for uniform R56 and
  1.214× for context R56, with large action errors. The >1.5× target fails there.
- At equal 198-row read budgets, dynamic count-weighted background pooling is
  slower and has worse mean action error than hard dynamic selection. The unit-
  weight tradeoff retains a gripper disagreement. Neither pooling nor unvalidated
  signal fusion is adopted. All negative controls and admission refusals remain.

## Twelve real episode attempts, then stop

The model and candidate were frozen before both cohorts. Development uses tasks
0/1/2 × initial state 1, Dense first. The additional fixed cohort uses tasks
3/4/5 × initial state 2, candidate first, without retuning. H100 exposure audit
found no previous reserved-identity outcomes, but historical full baselines cover
these tasks; this is not globally unseen confirmation. Both cohorts use seed 42,
`dreamwam-release-v1`, CPU OSMesa, 256×256 cameras, 32 returned actions, ten
denoising steps, ten executed actions per replan, 30 wait steps and 400 maximum
control steps. Only LIBERO determines success.

| Cohort | Dense | Candidate | Observed candidate − Dense | Nominal Wilson95, Dense / candidate |
|---|---:|---:|---:|---|
| Development, three pairs | 3/3 | 3/3 | 0 pp | [43.85,100]% / [43.85,100]% |
| Additional fixed, three pairs | 2/3 | 3/3 | +33.33 pp | [20.77,93.85]% / [43.85,100]% |

Dense task 5 reaches the 400-step limit: it is a genuine task failure, not an
infrastructure error. The candidate succeeds there in 93 steps. There are zero
errors, retries, incomplete outcomes or forced cleanups. Descriptively the two
cohorts contain Dense 5/6 and candidate 6/6, but this is not a benchmark or an
independent confirmation estimate. The task-stratified bootstrap correctly
withholds a paired interval because each task has only one pair. McNemar p=1
does not certify equivalence or the permitted five-percentage-point SR drop.

The exact effort ledger records **12 charged slots and 12 actual attempts**,
both reservations finalized. The other 38 slots under the user's hard cap of
50 remain unused. The study stops here regardless of these outcomes.

| Cohort / task | Dense success, steps, episode seconds | Candidate success, steps, episode seconds |
|---|---|---|
| Development / 0 | yes, 75, 25.520 | yes, 81, 25.890 |
| Development / 1 | yes, 101, 27.866 | yes, 114, 31.140 |
| Development / 2 | yes, 93, 27.547 | yes, 104, 30.240 |
| Fixed / 3 | yes, 79, 25.264 | yes, 85, 27.627 |
| Fixed / 4 | yes, 112, 33.355 | yes, 139, 37.644 |
| Fixed / 5 | no, 400, 100.901 | yes, 93, 28.342 |

Independent audits verify 135 development and 167 fixed-cohort artifact hashes,
every executed prefix against archived predictions, original action-layer budgets,
every native RGB read and zero numeric fallback events. All six initial state,
agentview and wrist input pairs have identical hashes. Later trajectories differ.

## Inference, IPC and episode time are separate

| Cohort | Warm model mean, Dense / candidate | Warm model ratio | Warm IPC-inclusive mean, Dense / candidate | All-call model ratio |
|---|---:|---:|---:|---:|
| Development | 286.64 / 131.58 ms | 2.178× | 295.78 / 139.94 ms | 1.922× |
| Fixed | 262.39 / 133.41 ms | 1.967× | 270.90 / 142.62 ms | 1.601× |

Warm means exclude each episode's first call and require a recorded prompt-cache
hit: 26/29 Dense/candidate calls in development, 57/30 in fixed validation.
Full distributions and cold calls are in the verification JSON. The first model
calls are 1.931/1.584 s in development and 1.646/1.592 s in fixed validation.
They exclude model loading but include lazy graph capture and prompt misses.

Total episode wall time is 80.933/87.270 s in development and 159.520/93.613 s
in fixed validation. The latter is dominated by Dense's failed 400-step episode;
on all five tasks where both succeed, the candidate takes more control steps
and longer wall time. Closed-loop timing ratios are descriptive, not controlled
same-input speedups. CPU rendering and shared GPU load are included where stated.

## Verification, provenance and replay

All model execution and tests ran on H100; source edits were committed locally,
pushed and fast-forwarded into clean server development trees. Runs used separate
immutable worktrees. No dependency, model weight, simulator or protocol version
changed; Python remains 3.10.20 and torch 2.7.1+cu126.

| Component | Exact revision / identity |
|---|---|
| Final model, offline screen and adapter | `41f515a5480cdcc87a18b147ab393e9cae337e9f` |
| Bounded controller | `b8477755bd413e1b435246e68bde3b949155d180` |
| Independent closed-loop auditor | `c6e11f90934db8da91c073708af7e179f9f0571d` |
| Development evaluator | `0f856c9ce160150286ed31f11001cca0acd04c32` |
| Fixed-cohort evaluator | `b5952a54369b8fcc11513c8508e5c7fdd3840ff2` |
| LIBERO | `8f1084e3132a39270c3a13ebe37270a43ece2a01` |
| Checkpoint SHA-256 | `6c087e5b9e201f19dbe92834b470a7ee18937a18ea9adcbb49886aa1e3ac4c61` |

The final real adapter check passes all 22 predictions/archived arrays, including
changing images/instructions, episode reset and cache close. Four actual BF16
CUDA graph modes pass, together with CPU invariants and the evaluator's complete
tests/schema checks. Existing adverse CUDA checks and resource admission refusals
are retained. Final config-only changes pass validate/doctor with evaluator code,
dependencies and schema unchanged. No model tests were rerun merely to update docs.

All large artifacts remain below H100
`/root/wenbiao_zhao/dreamwam-sr/outputs/dido-sparse-profile-20260920/`.
Frozen matrices/configs live in [experiment-plan.json](experiment-plan.json) and
the isolated action-eval branch's `dreamwam-dido-*-{dev3,fixed3}-osmesa.yaml`.
The two pair directories are `pilot3-native-va56-41f515a-shared50` and
`fixed3-native-va56-41f515a-shared50`; each controller preserves exact evaluator
commands, admission snapshots and fingerprints. Their audit directories are
`pilot3-native-va56-audit-c6e11f9` and `fixed3-native-va56-audit-c6e11f9`.

CPU-only replay on H100, with a fresh output directory:

```bash
base=/root/wenbiao_zhao/dreamwam-sr
cd "$base/.trees/dido-run-c6e11f9-audit"
export PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=''
py="$base/DreamWAM-fresh-6c52f36/.venv/bin/python"
pair="$base/outputs/dido-sparse-profile-20260920/fixed3-native-va56-41f515a-shared50"
out="$(mktemp -d "$base/outputs/dido-audit-XXXXXX")"
"$py" scripts/sparse/audit_dido_pilot.py --pair "$pair" --out "$out/integrity.json"
"$py" scripts/sparse/paired_sr.py --dense "$pair/dense/run" \
  --sparse "$pair/sparse/run" --label fixed_validation --json "$out/paired-sr.json"
"$py" scripts/sparse/summarize_policy_timings.py --dense "$pair/dense/run" \
  --sparse "$pair/sparse/run" --out "$out/policy-timings.json"
"$py" scripts/sparse/audit_initial_inputs.py --reference "$pair/dense/run" \
  --candidate "$pair/sparse/run" --out "$out/initial-inputs.json"
```

This replays existing evidence without model loading or new episodes. Historical
launch controls and the permanent ledger are documented in
[CLOSED-LOOP.md](CLOSED-LOOP.md); their presence does not schedule another run.

GPU 0 was not used. Offline admission required GPU 5 utilization ≤10% and
≥50000 MiB free; the frozen adapter/pilots explicitly allowed ≤50% under the
user's flexible-sharing authorization, retaining that memory floor. Sharing can
slow both projects. At closeout GPU 5 was at 0% utilization, 4 MiB used, with no
task-owned or matching frozen-run process. No foreign process was signalled,
dummy load started, training launched or historical queue resumed.

The evidence supports retaining uniform as the reference and treating native
selection/pooling as research variants. Another round would need a newly scoped
question: explain the initial-observation gripper divergence before tuning more
selectors, or obtain enough independently specified paired evidence to assess
SR. Author-only semantic claims require author data. None is silently converted
into more runs in this completed finite study.
