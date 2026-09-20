# H100 hybrid routing, dependency profiling and lower-budget pilots

Status (2026-09-20 UTC): **offline experiments and both KV25/KV18.75 pilots
COMPLETE**. Implementation is described separately
in [the executable V5 method](decision-support-method-20260920.md). These are
exploratory development results, not benchmark reproduction or SR non-inferiority.

## Scope, execution and audit

This round answers the user's requests to reduce retention, explain **which**
tokens are selected, and test the mechanisms in Introduction V5. It adds current
A→V and backward V→V support scores, separate read/refresh rankings, an explicit
structure-only execution mode, bounded grids, trajectory capture/export and
offline dependency interventions. It does not train/change weights or simulator
protocols. Adaptive refresh is not yet implemented or validated.

Seven finite replay cohorts completed **999/999 timed requests**, covering
**57 configuration/dataset cases, 51 unique policy identities**. An independent
post-run audit reloaded every accepted action and eager reference, checked
artifact SHA256 and **byte equality including signed zero**, full request
coverage, all 300 action-layer updates and same-invocation control pairing.
No latency outlier was removed. Complete table:
[results.csv](evidence/hybrid-routing-20260920/results.csv).

| Cohort | Cases | Inputs | Timed requests |
|---|---:|---:|---:|
| Original action-drift budget curve | 6 | 3 historical observations | 108 |
| Six selectors × four KV budgets | 24 | same 3 | 324 |
| No Sparse refresh / each single step 1–9 | 10 | same 3 | 135 |
| Structure-only: three selectors × three budgets | 9 | same 3 | 108 |
| Lower budgets / selectors on Dense trajectories | 8 | 9 snapshots, 3 episodes | 324 |

Three historical observations are Spatial tasks 0/4/8, initial state 0, already
exposed in earlier calibration. New observations are first/middle/last policy
inputs from each of three Dense trajectories on tasks 0/1/2, initial state 1.
They are hash-checked against the actual policy inputs, include full images/state,
and contain no success labels or teacher actions. Nine snapshots are **not nine
independent episodes**. None of these inputs is labelled held-out confirmation.

All GPU execution uses H100 policy GPU 3
(`GPU-c0af33a9-498c-ff7c-bb56-e9992ccded30`), with fresh admission and an unused
available card among authorized 3/4/5. The host is shared, not reserved. One
structure-cohort launch was deferred when there was no admissible spare; it
made no model attempt and later ran after a new admission check. Other users'
processes were not signalled. Closed loops use private CPU OSMesa, not EGL.

Environments remain Python 3.10.20 / torch 2.7.1+cu126 / CUDA 12.6, original
robosuite 1.4.1 / MuJoCo 3.3.2. No installs, dependency upgrades or lock changes.
Checkpoint SHA256:
`6c087e5b9e201f19dbe92834b470a7ee18937a18ea9adcbb49886aa1e3ac4c61`.
Code/config changes were locally committed/pushed, then cleanly pulled on the
server. Active run worktrees remained frozen.

## Selection matters, but the proposed attention proxy is not validated

The old selector conflates read value and refresh urgency. New `action` uses
current first-layer A→V mass; `action_context` adds the keys supporting its top
visual query seeds via V→V, and refresh urgency additionally uses token drift.
All have balanced per-frame quotas; all action keys remain in joint softmax.
Projection/probe, route, packing and cache costs are inside full-request timing.
Separate instrumentation traces are not used as latency samples.

The following are mean final executable-action relative L2 errors against Dense,
**not SR**. D at step 0, S at step 5, Q=10%; only selector/KV budget varies.

| Selector | KV12.5 | KV25 | KV37.5 | KV50 |
|---|---:|---:|---:|---:|
| uniform | 1.4760 | 1.4495 | 0.0975 | 0.0814 |
| drift | 1.4392 | **0.1084** | 0.8773 | 0.0902 |
| legacy action_drift | 1.5111 | 0.8307 | 0.1580 | 0.1377 |
| visual_context | 1.2424 | 0.6707 | 0.1868 | 0.1679 |
| current action | 1.2799 | 1.2506 | 0.2010 | 0.1397 |
| action_context | 1.4638 | 0.9930 | 0.1717 | 0.1432 |

Every cell is retained. Read error is not monotonic in budget because selected
positions, refresh queries and nonlinear trajectories change. The extra context
probe does **not** consistently outperform simple selection. At KV25, simply
changing from old action-drift to drift reduces this open-loop error strongly;
that is not evidence for the proposed context mechanism.

The original action-drift curve gives 1.904× at KV12.5 but error 1.5111;
KV37.5 gives 1.874× / 0.1580; KV75 gives 1.825× / 0.0814. Thus speed alone
would select poor low-budget candidates. Prior KV25 action-drift failures remain
in [the earlier pilot record](hybrid-pilot3-osmesa-20260920.md), not overwritten.

## Measured refresh positions, not a hand-picked step

For KV25/drift/Q10, the frozen search enumerates zero Sparse refreshes and each
single refresh at steps 1–9. No-refresh gives action error **0.1054**, versus
0.1131, 0.1110, 0.1100, 0.1119, 0.1084, 0.1076, 0.1120, 0.1359 and 0.1811
for steps 1 through 9. Its warm mean is **123.76 ms, 2.119×** matched Dense.
This justifies testing no-refresh on these inputs, not claiming a universally
optimal schedule or an adaptive trigger. Shared-load timing spikes in the
step-5/6 group remain in the table and its contemporaneous controls.

With zero Sparse refreshes, `selection: drift` uses **uniform selection at the
Dense anchor**, because no previous visual input exists. No later drift ranking
runs. The selected method is therefore static uniform support plus feature
reuse, not online drift-aware or action-aware routing. Q=10% is an inactive
Sparse-step setting in this particular schedule. It computes one full visual
anchor, followed by nine action updates using packed visual K/V.

## Structure reuse is not the source of the approximately 2× result

The separate structure-only implementation retains indices but recomputes all
selected current visual Q/K/V, attention, FFN and residuals on every step. It
never passes old K/V to its transformer. Unselected visual rows bypass from the
current input. D at step 0 / route rebuild at step 5 is held fixed here.

| Q=KV budget | Uniform speedup | A→V speedup | A→V + context speedup |
|---|---:|---:|---:|
| 10% | 1.131× | 1.129× | 1.124× |
| 25% | 1.104× | 1.102× | 1.100× |
| 37.5% | 1.064× | 1.062× | 1.061× |

None meets >1.5×. Action errors also remain substantial. A faster feature-cache
result cannot be cited as validation of V5's sentence “reuse the sparse structure,
recompute current Q/K/V.” Both mechanisms are exposed independently so this
distinction can be ablated rather than hidden.

## Dependency interventions and cross-step stability

**153/153 interventions** completed: 45 remove equal-sized balanced visual key
sets at steps 1/5/9; another 108 target **future keys only**, comprising 45 score
comparisons plus 63 disjoint-group interventions (7 groups × 3 steps × 3 inputs).
Observed-frame keys are untouched in the latter cohort. Original tensor sizes,
heads, remaining masks and all model/sampler steps remain unchanged: this is
causal perturbation analysis, **not a sparse timing measurement**.

Average future-only removal effects:

| Removed keys | Action rel. L2 | Executed-prefix rel. L2 | Future-latent rel. L2 |
|---|---:|---:|---:|
| uniform | 0.00980 | 0.00559 | 0.05629 |
| highest A→V | 0.00537 | 0.00364 | 0.02912 |
| lowest A→V | **0.01292** | **0.01068** | **0.08119** |
| visual-context | 0.00622 | 0.00415 | 0.04654 |
| A→V + context | 0.00537 | 0.00364 | 0.02912 |

Highest A→V and A→V+context select identical top future sets in these probes.
Removing lowest A→V scores has larger average effects. This directly cautions
against using **this first-layer attention score** as a causal decision-value
oracle; it does not prove that every action-attention signal at every depth fails.

Across 63 future-group interventions, pooled action/latent sensitivity Pearson
correlation is **0.8382** (descriptive; correlated inputs/stages). This coarse
study does not establish a strong generation/decision separation. Future-latent
distance to Dense is not decoded-video quality, true prediction error or SR.
The existing Head×Stage classification is a different granularity and cannot be
silently substituted as validation of this token-routing proxy.

On untouched Dense trajectories, adjacent-step support Jaccard averages 0.8780
for A→V and 0.8899 for A→V+context. Separating frames matters: A→V+context is
**0.9865 observed-frame vs 0.8499 future-frame**; visual-context alone is 0.9798
vs 0.7097. This is limited support-stability evidence at layer 0, not proof that
cached values remain valid or that a particular refresh threshold is safe.

## Trajectory expansion and matched closed loops

Both closed-loop arms retain `dreamwam-release-v1`: max 400, wait 30, 256²,
replan 10, seed 42, action horizon 32, ten denoising steps, fixed-per-predict RNG.
The stronger Dense baseline uses exact prompt and conditioned-frame reuse with
transformer graphs. The candidate uses the same prompt cache/backend. Each pair
has tasks **0/1/2 × initial state 1**. This is development screening; repeated
Dense outcomes in the follow-up cannot be counted as independent new episodes.

KV25 no-refresh completed at **14:09:59 UTC**, controller exit 0,
`PILOT_PAIR_COMPLETE`, no errors/retries/forced cleanup. Dense succeeds at steps
75/101/93; candidate at **83/115/104**. Both are **3/3**. First-input state and
both cameras match byte-for-byte on all three pairs. Native rendering completed
748/748 Dense and 814/814 candidate reads.

Warm request means are **262.25 ms Dense vs 131.91 ms candidate, 1.988×**;
IPC-inclusive warm speedup is 1.928×. Warm excludes the first call of each
episode and requires a recorded prompt-cache hit. All-call model means,
including cold setup, are 316.68 vs 172.50 ms (1.836×); these unequal trajectory
lengths make the ratio descriptive. Whole episodes are **27.48 vs 29.41 s**,
so there is no rendered end-to-end acceleration. The candidate executes more
environment steps. Realized read support is **74/294** rows (nominal 25%).

Each 3/3 Wilson95 interval is [43.85%, 100%]; no SR-preservation claim follows.
Bootstrap intervals are withheld with one initial state per task. All 32
candidate requests retain 300 action-layer updates; feature reuse is confined
to the current inference request, never across observations.

The nine exported Dense snapshots are independently replayed under eight
lower-budget/selector cases. Uniform-anchor no-refresh results:

| Nominal KV | Warm time | Matched speedup | Action rel. L2 | Prefix-10 rel. L2 |
|---|---:|---:|---:|---:|
| 12.5% | 124.46 ms | 2.120× | 0.3583 | 0.4477 |
| 18.75% | 123.66 ms | 2.123× | 0.2603 | 0.3354 |
| 25% | 124.43 ms | 2.107× | 0.2450 | 0.3113 |
| 37.5% | 126.94 ms | 2.068× | 0.2366 | 0.3006 |

Differences in latency between 18.75% and 25% are tiny, not a demonstrated extra
acceleration. The 18.75% pilot is a single bounded follow-up to locate a retention
boundary; no further descending-budget rollout is planned in this round.

### KV18.75 follow-up: lowest retention with a completed pilot this round

Completed at **14:24:46 UTC**, controller exit 0, `PILOT_PAIR_COMPLETE`, no
errors/retries/forced cleanup. Both arms again succeed **3/3**. Candidate steps
are **85/127/107**, versus Dense 75/101/93. Nominal 18.75% realizes **56/294
visual keys = 19.05%**, distributed [19,19,18] across the three latent frames.
This is the post-anchor visual read budget, not total attention density or total
inference work; all action keys and the full first Dense step remain.

Warm model means: **266.49 vs 128.29 ms, 2.077×**. IPC-inclusive warm:
276.25 vs 136.77 ms, **2.020×**. All-call model means including cold/setup:
322.80 vs 175.08 ms, **1.844×**. Whole episodes: **28.11 vs 30.10 s**, again no
rendered end-to-end gain. Warm sample counts are 26 vs 30 (29 vs 33 total calls);
their trajectories differ, so the separate nine-input 2.123× replay is the
controlled same-observation timing check, not these descriptive closed-loop ratios.

Initial state and both-camera hashes match 3/3 pairs. Native reads complete
748/748 Dense and 848/848 candidate. Repeated Dense actions across KV25/KV18.75
are byte-identical on all three episodes. They are repeatability evidence,
**not six independent Dense outcomes**. The same wide 3/3 Wilson intervals and
withheld task bootstrap apply. All 33 candidate requests execute one Dense,
zero Sparse and nine Reuse steps, with 300 action-layer updates and 56-row
post-anchor support. No lower-budget SR pilot was run.

Retain KV18.75 as the lowest-budget **small-pilot candidate**, with KV25 as a
nearby control. Neither is a certified minimum or a demonstrated SR-preserving
method. Extra reduction from KV25 did not yield a substantial same-input speed
gain, and the candidate requires more simulator steps.

## Provenance and reproduction

H100 artifact root: `/root/wenbiao_zhao/dreamwam-sr/outputs/hybrid-routing-profile-20260920/`.
Every timing directory contains immutable source/input/config/hardware identity,
full commands/attempts, eager references, all request actions and journal. The
independent `study-audit.json` covers all 999 requests; `study-results.csv` is
the checked-in table's source. `dependencies/` and `future-dependencies/` retain
all interventions and reference actions/latents. `trajectory-dense/manifest.json`
records each source observation/hash and deterministic first/middle/last rule.
The checked-in [evidence index](hybrid-routing-evidence-20260920.json) hashes the
manifests, full journals, reports, table, adapter checks and paired-run provenance.

Source lineages: budget replay `cef5050`; selector replay `ea2d850`; refresh,
initial structure and all-frame intervention `c07c83c`; trajectory, future-only
intervention and remaining structure `ea30842`. Closed-loop model stays frozen
at `c07c83c`; evaluators are `e5a5cc5` (KV25) / `3a02f89` (KV18.75). Later
reporting/default-validation commits do not alter these completed/live sources.

Representative server commands (full resolved invocations are in manifests):

```bash
python scripts/sparse/generate_hybrid_schedules.py \
  --candidate-sparse-steps 1:10 --refresh-counts 0,1 \
  --recompute-ratio .1 --read-ratio .25 --selection drift \
  --backend cuda_graph --out refresh-grid.jsonl
python scripts/sparse/benchmark_hybrid_schedules.py \
  --schedules refresh-grid.jsonl --inputs "$INPUT_MANIFEST" --split-label debug \
  --reps 3 --warmup 2 --group-size 2 --controls dense_strong \
  --allow-shared-gpu --trace --out-dir "$OUT/refresh-grid"
python scripts/sparse/profile_hybrid_dependencies.py \
  --inputs "$INPUT_MANIFEST" --steps 1,5,9 --scope future --group-count 7 \
  --out-dir "$OUT/future-dependencies"
python scripts/sparse/export_trajectory_inputs.py \
  --run "$OUT/pair25nr/dense/run" --out-dir "$OUT/trajectory-dense"
```

Before each paired rollout, committed configs pass `validate`/`doctor`, and
real-checkpoint adapter checks compare 14 requests (seven per arm) with their
own uncached eager outputs, changed observations, prompt misses/hits/reset and
released graph/cache state. The bounded controller requires matching effective
fingerprints, unchanged protocol, native-read guarding and fresh GPU admission.
The config/adapter skills keep these safeguards separate from model selection
and benchmark-owned success. No 50-pair or old 500-episode queue was started.

Final server verification: **341 CPU tests passed, 18 CUDA-only tests skipped,
16 subtests passed**; the actual H100 graph/structure test selection passed
**14/14**, followed by real-checkpoint eager checks for all measured variants.
action-eval passed **201 tests**, with generated schema byte-equal to the checked-in
schema. The final controller-default fix has its own regression test; active
pilots used their original frozen controller, with already explicit effective
options. Some preflight/report commands initially had an interpreter path, a
detached-pull invocation, or a nonexistent test-directory argument; these were
corrected before the corresponding work and never counted as task outcomes.

All owned model/controller jobs exited normally; policy GPU 3 was released.

The next scientific task is a better **causally calibrated** decision-value
signal, potentially reusing the existing layer/head sensitivity interface,
followed by context routing on broader development trajectories and a frozen
confirmation set. Do not describe the current first-layer proxy as established
decision sufficiency or the explicit no-refresh schedule as adaptive refresh.
