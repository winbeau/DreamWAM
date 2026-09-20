# Native DreamWAM action–video evidence

Status: raw capture and descriptive replay **verified**, 2026-09-20 UTC.
The full goal remains active. The bounded diagnostic follow-up is also complete;
it does not establish a consistently better selector, a speedup or SR preservation.
The subsequent [400-call online screen](ONLINE-RESULTS.md) is recorded separately;
it tests actual selection/packing cost and retains two development candidates.

## Cohort, execution and provenance

These are **our self-captured DreamWAM observations and native tensors**, not
DIDO author demonstrations, semantic boxes/tracks or attribution records.
The user has no private author-data link. See [source audit](SOURCES.md).
The nine observations are three positions in each of three historically exposed
Spatial development episodes (tasks 0/1/2, initial state 1). They are correlated
within trajectories and are not held-out confirmation or nine new rollouts.

Capture source `025be1235d29f4227ba0311e22daabb00ad1e72b`, H100 detached tree
`.trees/dido-run-025be12-profile`; checkpoint SHA-256
`6c087e5b9e201f19dbe92834b470a7ee18937a18ea9adcbb49886aa1e3ac4c61`.
Python 3.10.20 / torch 2.7.1+cu126, existing environment unchanged. The user
authorized sharing GPU 5 with their other project, superseding its historical
last-free-card restriction. GPU 0 was reclaimed and remained unused by this task.

At admission GPU 5 used 2,937 MiB, utilization 0%; another project's PID 385222
was left untouched. Capture ran **18:19:09.722–18:22:09.210 UTC**, exit 0,
18/18 predictions. The predeclared sampling covers all 10 denoising steps,
layers `[0,9,19,29]`, heads `[0,6,12,18]`, all 294 video cells and 32 action
tokens. This is a fixed sample of heads/depths, not exhaustive model coverage.
All nine inputs pass **bitwise executable-action, pre-binarization raw-action
and final-video-latent parity** against independent native controls.

Raw archive: 567 compressed NPZ files, 1,438,891,488 uncompressed array bytes
(within the 2 GiB cap), 785,128,759 compressed bytes. Independent CPU replay
ran **18:22:35.421–18:22:58.347 UTC**, exit 0: all 567 files validated,
360 attention cells, 423,360 token-score rows, 2,880 equal-budget routes,
12,960 head comparisons and 33,264 regional/global transitions.
The GPU process exited after capture. None of these calls is a closed-loop
episode; closed-loop attempts remain **0/50**.

All artifacts are under H100
`/root/wenbiao_zhao/dreamwam-sr/outputs/dido-sparse-profile-20260920/`:

| Artifact | SHA-256 / role |
|---|---|
| `native-025be12/report.json` | `210642d5ec3a4cfff860335e18fd38ec4d2ddbb65320f711bcae88e463f0c9f7` |
| `native-025be12/raw/records.jsonl` | `3cb63e40c10984a96cc241aeb5a9bd650500028bb8774b265fa2050738165dc3`; complete per-NPZ hashes/schema |
| `analysis-025be12/report.json` | `142c72009a8f1f827fb01d90aa9ad11f2da598f3ead64daabf0bf54626cb95c3`; hashes all four full exports |
| `summary-b3d4160/report.json` | full numeric summary, generated at `b3d4160e641caf4eb41ec19634c03ae8a4f9608a` |
| `summary-b3d4160/support-agreement.png` | `34c7463e2e682455f2e1f17f2a05e4b2e31ab346ba03fa2bd6d4d1fbdd6635b8` |
| `summary-b3d4160/support-agreement.pdf` | `b7b79f9b17864474e19ecca99293b60bdfb1293184f353e5ae83601d8bf8abf0` |

## What the native profile actually shows

Every route has exactly 56 cells, balanced `[19,19,18]` across frames. Scores
average the four sampled heads before selecting the route. Jaccard is
intersection/union of selected original positions. The table contains descriptive
means, without treating repeated cells or correlated observations as trials.

| Proxy | Adjacent denoising steps | Consecutive sampled depths | Pairwise sampled heads |
|---|---:|---:|---:|
| A→V | 0.896 | 0.150 | 0.175 |
| A→V × value norm | 0.890 | 0.190 | 0.173 |
| Video-time value difference | 0.915 | 0.427 | 0.559 |
| Backward VV support seeded by A→V | 0.877 | 0.139 | 0.150 |

The first column averages 324 transitions per proxy; the second 270 sampled
depth transitions; the third 2,160 sampled-head pairs. These are **not**
independent sample sizes or confidence intervals. Depth gaps are 0→9, 9→19,
19→29, not neighboring transformer blocks.

Within a fixed sampled depth, action-based supports vary less across neighboring
denoising steps than across depths or heads. This motivates testing depth-specific
signals; it does not prove that early action noise is the failure mechanism, or
that stable selected positions permit stale value reuse. Both claims require
separate action/latent interventions and real execution measurements.

The dynamic score is **identically zero in the observed frame**, so its stable
tie-breaker selects the same first 19 observed cells. This mechanically raises
all-frame agreement. Between step 0/layer 0 and step 9/layer 29, its mean Jaccard
is 0.298 across all frames but only **0.100 on future cells**. For A→V these
values are 0.134/0.147, and for A→V×value norm 0.119/0.118. These endpoint
comparisons change depth and time together; do not attribute them to either
axis alone. Future-only statistics prevent treating current-frame ties as a
discovered dynamic prior.

A→V×value norm overlaps A→V at mean Jaccard 0.633 (0.610 on future cells);
dynamic selection overlaps A→V at 0.155 (0.134 on future cells). The proxies
therefore select materially different sets. No causal retention advantage
follows from those differences, and no fusion weights have been fitted.

## Frozen diagnostic follow-up and remaining gates

The [experiment plan](experiment-plan.json) fixes a **49-prediction** diagnostic
study on two existing middle-trajectory observations: 47 interventions and two
fresh native controls, zero rollouts. It includes AV, VV and joint key deletion;
denominator-preserving zero-value replacement; previous/current K/V recompute
counterfactuals; uniform/random/action/value/dynamic/context controls; and
early/shallow versus final/deep endpoint checks. All comparisons retain the
same within-operation budget of 56 cells. A partial design is intentional;
it is not an exhaustive factorial or an SR benchmark.

At `1cb9370`, 14 server CPU study/intervention/pooling tests passed. The first
launch failed its second admission check **before model loading**, exit 1,
with **0 attempted predictions and 0 raw artifacts**. Its report/log and frozen
cases remain in `interventions-1cb9370/`. The rejected snapshot was not included
by the old exception, so its exact reason is unknown. `89bd1ea` records the full
rejected inventory and checks actual `memory.free`; no utilization threshold was
relaxed. The same-budget retry at `89bd1ea` completed **18:36:14.986–18:37:01.122
UTC**, exit 0, with **49/49 predictions and 47/47 intervention cases**. Both new
Dense controls match the earlier native capture bitwise. All 49 raw artifacts
(11,151,616 bytes) and all 47 recorded output diagnostics were subsequently
replayed from their actual arrays. Peak allocated/reserved GPU memory was
25,113,829,888 / 25,249,710,080 bytes. The owned model process exited normally.

| Diagnostic artifact | SHA-256 |
|---|---|
| `interventions-89bd1ea/report.json` | `b466774cb73d0c9752bb57f78799b95ccdc19354fefbf465903ed593ac79340d` |
| `interventions-89bd1ea/frozen-cases.json` | `368c1154ec0accfe88c2e692f4c7db277e62c0a04e70dac1a85191129e24165e` |
| `interventions-89bd1ea/raw/records.jsonl` | `90be74b1396ebe2bc71f4a7ed4e4f9f7a9b7cedb7859706ba355c3b95c525231` |
| `intervention-audit-3894ef1/report.json` | `0d99cb8b306fc3beaf15ebccdc325c10e868d223f2d6c8a42d1c368c448500c1` |
| `intervention-audit-3894ef1/case-metrics.csv` | `75ead146a8518b98919de9e3b240984a3363fc7ab7d091a447b90ac1d68192cb`; complete 47-case table |

The results distinguish direct reads from video-side effects:

- Every AV-only intervention preserves the final video tensor bitwise. At the
  last denoising step and final layer, VV-only deletion preserves the complete
  raw action tensor bitwise while changing future latents by relative L2
  **0.0724** (uniform) or **0.1021** (dynamic). Video error therefore cannot be
  substituted for action impact, even on these actual checkpoint inputs.
- At step 5/layer 19, VV deletion in the second observation gives executed-prefix
  raw-action relative L2 **0.00434** (uniform), **0.00969** (value-aware action),
  **0.01089** (dynamic) and **0.00797** (action-seeded context). In the first
  observation the dynamic value is **0.00378**, below uniform **0.00415**.
  A large effect in one observation is not a stable ranking across the cohort.
- On the first observation, AV value replacement gives raw-prefix relative L2
  **0.00546** for value-aware action versus **0.00365** for uniform. Deletion
  renormalizes attention whereas this replacement preserves its denominator;
  the operations probe different mechanisms and their magnitudes are kept
  separate in the full table.
- The limited AV recompute control gives raw-prefix relative L2 **0.00296**
  (uniform), **0.00322** (value-aware action) and **0.00316** (dynamic). This
  particular update choice does not show either new ranking beating uniform.
- No executed-prefix gripper sign changes occur in the 47 cases. These are
  single-cell interventions on two exposed observations, so that observation
  provides no closed-loop success guarantee.

All variations and unfavorable results remain in the raw report. The reproducible
CPU auditor exports every case rather than selecting favorable ones:

```bash
CUDA_VISIBLE_DEVICES='' python scripts/sparse/audit_action_video_interventions.py \
  --study /path/to/interventions-89bd1ea --out-dir /new/path/to/diagnostic-audit
```

New real-checkpoint predictions so far: **67** (18 capture + 49 diagnostics;
the refused launch attempted zero). New closed-loop episodes: **0/50**.
The scripted audit completed at **18:50:06 UTC**, exit 0; all native-control
parity, execution traces and 47 metrics reproduce from the archived arrays.
At that time all owned model PIDs were absent and GPU 5 was at 4 MiB / 0%.

Online scoring/packing, feature versus structure reuse, background pooling,
explicit refresh scans, CUDA graph parity, matched full-predict timings,
adapter checks and bounded paired SR remain open. Historical uniform feature
reuse at roughly 2× is retained as a control, not claimed as a new selector
result. The SR margin is 5 percentage points; the global closed-loop cap is
50 attempts across Dense and all candidates, with no automatic expansion.

Reproduce using [PROFILE.md](PROFILE.md), the exact source revisions above,
the immutable input manifest in the plan and the unchanged H100 environment.
Large raw arrays remain outside Git. Every subsequent result must retain its
own revision, frozen cases, errors, hashes and resource-release record.
