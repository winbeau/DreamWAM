# Offline Head × Stage sensitivity calibration

Status: **VERIFIED offline collection, classification and combined allocation
comparison; M1 speed/SR unverified**. Updated **2026-09-20 UTC**. The complete
sweep ended **03:41:50 UTC**, its classification/audit passed at **03:42 UTC**,
and combined inference completed **03:44:38 UTC**, all exit 0. The user
explicitly requested offline action/video sensitivity classification and a
trial. The frozen SR candidates are unchanged; this does not certify a budget.

At the same 50% future-key mask budget, the separate check input's normalized
action relative L2 falls from **0.153968 uniform** to **0.082719 head-only**
and **0.084239 Head × Stage**. Thus sensitivity-based allocation helps this
offline proxy, while **Stage has not shown an additional action advantage**
over head-only. The latter comparison is preserved rather than selecting only
the favorable uniform baseline. This is not a success-rate or speedup result.

Complete artifacts: [2,160-row type/metric CSV](evidence/head-stage-20260920/head-stage.csv),
[classification summary](evidence/head-stage-20260920/classification-summary.json),
[sensitivity heatmap PNG](evidence/head-stage-20260920/head-stage-sensitivity.png)
/ [PDF](evidence/head-stage-20260920/head-stage-sensitivity.pdf),
[cross-input stability PNG](evidence/head-stage-20260920/head-stage-stability.png)
/ [PDF](evidence/head-stage-20260920/head-stage-stability.pdf),
[full audit](evidence/head-stage-20260920/full-audit.json),
[combined profiles and raw actions](evidence/head-stage-20260920/combined-report.json),
[combined audit and executed-prefix metrics](evidence/head-stage-20260920/combined-audit-and-prefix10.json).

The earlier fast visual-cache policies use uniform refresh cadences and, in the
guided variant, an AV-weighted token-drift score. They have no measured
Head × Stage profile. Their speedups therefore do not establish the paper's
complete M1–M3 chain; see the [mechanism audit](paper-mechanism-status-20260920.md).

## Fixed experiment

The collection freezes three real LIBERO Spatial observations: tasks 0, 4 and
8, initial state 0, seed 42, after the original 30 settle steps. The evaluator
supplies both 256 × 256 uint8 camera images, proprioception and the task
instruction. A renderer-only process captures them before the model process
starts. Every paired intervention subsequently reads the same hash-verified
NPZ bytes. [The captured inputs](evidence/head-stage-20260920/input-contact-sheet.png)
were visually inspected; both camera views contain the expected robot/table
scene and task-dependent object arrangements.

These official identities are now **exposed calibration inputs**, not held-out
SR evidence. Tasks 0/4 define the relative types and task 8 provides a separate
cross-input check. This is a small, related set of initial scenes, with no
mid-episode states or other suites; it cannot establish generalization alone.

The model remains Joint bf16, checkpoint SHA256
`6c087e5b9e201f19dbe92834b470a7ee18937a18ea9adcbb49886aa1e3ac4c61`,
30 layers, 24 heads, ten denoising steps, horizon 32, replan 10 and the
released RNG behavior. There are 294 video tokens (98 conditioned, 196 future)
and 32 action tokens. Environments are reused unchanged: Python 3.10.20,
torch 2.7.1+cu126; no installation, lock change or sync was performed.

Each intervention changes exactly one layer/head during one fixed stage:
`[0,3)`, `[3,7)` or `[7,10)`. It removes that head's future-video-query to
future-video-key connections. Conditioned-frame rows/keys and all action
rows retain the native mask; every other head retains the native mask. At
this strong `future_keep_ratio=0` setting, the target head's total allowed VV
pair density is **3/7**, because conditioned-frame structure remains. This
is not zero visual computation or a timing claim.

The original fused joint attention still executes, using the same contiguous
expanded mask layout in restricted and full-budget controls. Each intervention
continues the complete sampler to final normalized/denormalized actions and
future video latents. No temporal reuse, Graph, AV selection, FFN optimization,
training or online budget adjustment is active. Raw actions and continuous
action/video relative L2 metrics are retained, including translation, rotation
and gripper diagnostics. There are **2,160 units × 3 inputs = 6,480 interventions**.

## Controls and classification rule

- Native repeated actions and the full-budget probe must be bitwise identical
  before collecting each input. A full-budget control after every whole layer
  must reproduce the original action and video reference.
- Every intervention must execute 300 layer-attention calls, ten video
  scheduler steps and exactly its declared target steps. The conditioned
  latent frame must remain bitwise identical. Non-finite values or drift abort
  the collection; weights' version counters and evaluation options are checked.
- The first two inputs' mean normalized-action/future-video relative L2 sets
  separate empirical 25th/75th percentiles. Above both upper quartiles is
  `mixed_sensitive`; above only action/video is `action_sensitive` /
  `video_sensitive`; at or below both lower quartiles is `low_impact`; the
  remainder is `intermediate`. These are **relative types**, not natural
  clusters, pure semantic functions or SR-safe labels.
- The third input uses the same thresholds. Report label agreement, the full
  confusion table, action/video rank correlations and results within each
  stage. Keep continuous values in the CSV. The updated heatmap uses a common
  scale across stages for each metric and actual layer/head tick labels.
- Stage lengths are 3/4/3, so raw stage contrasts also reflect intervention
  dose. A single severe restriction does not establish a dose-response curve;
  low-impact interventions may interact when combined.

The classification rule was committed before the full sweep. The later
analysis update adds provenance, within-stage summaries and optional attention
statistics without changing this rule or its thresholds. No SR tolerance has
been chosen; the user will decide after complete Pareto results.

## Verified preparation and complete collection

Source `ea5ac5f` supplies the probe/capture/collector; `7c79c96` adds the first
classifier. Tests on the server passed **54/54**, exit 0, covering native
full-budget parity, mask scope, unaffected action/other-head rows, stage
boundaries and cleanup. The real-input capture completed **02:45:34–02:45:49
UTC**, exit 0, on physical GPU 4 after validate/doctor passed for the frozen
Spatial config.

The real-checkpoint smoke test at `7c79c96` completed **02:48:36–02:49:35 UTC**,
exit 0: layers 0/29 × heads 0/23 × three stages × three inputs = **36
interventions**. All three native/full-budget controls passed. Its 12-unit
classification is a pipeline check, not the full M1 result.

The full `7c79c96` sweep began **02:50:43.732 UTC** on GPU 4
(`GPU-490b4a76-6210-31b9-4e03-838a113cf5f4`), collector PID 330934. At the
03:06 audit, 1,872 of 6,480 interventions were journaled and the collector was
still running. It subsequently completed all 6,480 interventions at
**03:41:50.573 UTC**, exit 0. All three full-budget controls and all 90
end-of-layer Dense drift checks passed; the collector verifies unchanged
parameter versions/settings before marking completion. Physical GPU 7 was
deliberately unused by this task; it later became another evaluation's
dedicated rendering card. Free memory does not imply a reservation, and
shared-host load is recorded. Separate SR runs retain their frozen worktrees.

Server artifact root:
`/data/chenjiayu/wenbiao_zhao/dreamwam-sr/outputs/head-stage-calibration-20260920`.
The run is `full-7c79c96/`, with `manifest.json`, `interventions.jsonl` and
three `reference-*.npz` arrays. The outer wrapper records collector and
classifier exit codes separately. Classification refuses incomplete sweeps.

```bash
# In the pinned DreamWAM-head-stage-7c79c96 worktree; existing environment.
PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  scripts/sparse/collect_head_stage_impact.py \
  --inputs "$ARTIFACT_ROOT/inputs-gpu4/manifest.json" \
  --out-dir "$ARTIFACT_ROOT/full-7c79c96"
```

## Dense AV/VV observations

An additional observation-only pass at **`f489335`** completed
**03:05:34.580–03:06:04.375 UTC**, exit 0, on initially empty GPU 0
(`GPU-0b850a78-f1db-6bda-8dd5-d96296bc1bfc`). It covers the same three
inputs and all heads/steps: **21,600 step/head records**, aggregated to **6,480
input/layer/head/stage records**. All three instrumented final actions,
normalized actions and video outputs remained bitwise equal to their controls.
The server tests for this addition and the probe passed **9/9**, exit 0.

The original fused attention produces all model outputs. Separate float32
Q/K calculations respect the native joint mask and record mean A→conditioned,
A→future and A→action mass; future-V→conditioned/future/action mass; entropy
of normalized aggregate future-key distributions; and AV/VV distribution
cosine. Entropy is aggregate-key entropy, not mean per-query entropy. These
statistics describe attention; they do not replace causal sensitivity or
prove the AV–VV bridge helps.

An independent record audit found 21,600 unique identities, finite metrics,
maximum joint-mass sum error 3.33e-7 (AV) / 1.79e-7 (VV), and exactly zero
V→action mass. The final classifier joins statistics only when checkpoint,
input-manifest hash and all identities match. Type labels remain unchanged.

```bash
# Pinned DreamWAM-head-stats-f489335 worktree, same external checkpoint/env.
PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  scripts/sparse/collect_head_attention_stats.py \
  --inputs "$ARTIFACT_ROOT/inputs-gpu4/manifest.json" \
  --out-dir "$ARTIFACT_ROOT/attention-f489335"

# Run only after complete sensitivity collection; a separate analysis folder
# preserves the original 7c79c96 classifier output.
.venv/bin/python scripts/sparse/classify_head_stage.py \
  "$ARTIFACT_ROOT/full-7c79c96" \
  --attention-dir "$ARTIFACT_ROOT/attention-f489335" \
  --out-dir "$ARTIFACT_ROOT/analysis-f489335"
```

The completed `f489335` analysis joined all 6,480 attention records by exact
identity and left the original `7c79c96` classification unchanged. The
`90b3b77` audit verified the complete matrix, target steps/masks, finite
outputs, control flags and every class assignment. Recomputing all recorded
denormalized action metrics from raw actions gave **zero discrepancy**. The
intervention JSONL SHA256 is
`5d4a854d5024b683ad94ad0e8ea82f42aded13a83a131bb923e067b02f4cdb10`.
The updated full classification SHA256 is
`0321feb48e6d43085f3294a5b1329c8df50ab12b612b7ef7c51d7a49177e539f`.
The large raw matrix and reference arrays remain under the server artifact
root; small results and exports are committed here. Both figures were viewed.

## Measured relative types and stability

| Relative type | Fitted units | Same type on check input |
|---|---:|---:|
| Action-sensitive only | 81 | 55 / 81 |
| Video-sensitive only | 81 | 12 / 81 |
| Mixed-sensitive | 459 | 268 / 459 |
| Low impact | 324 | 65 / 324 |
| Intermediate | 1,215 | 972 / 1,215 |
| Total | 2,160 | 1,372 / 2,160 (**63.52%**) |

The fitted action quartiles are 0.00344428 / 0.00460675, and the video
quartiles 0.00752477 / 0.01851257. The two sensitivity metrics correlate
strongly (Spearman **0.839**), so this result does not show a clean semantic
split into action and video heads. Marginal-chance label agreement is 41.67%,
with Cohen's κ **0.375**. Cross-input action/video rank correlations are
**0.639 / 0.792**. Within early/middle/late stages, action correlations are
**0.364 / 0.663 / 0.833**; video correlations are **0.742 / 0.844 / 0.971**.
Hard relative types therefore have limited stability, especially early.

The low-impact category deserves a precise interpretation: 259 of its 324
units move to the intermediate category, but **none becomes upper-quartile
action- or video-sensitive** on this check input. Low exact-label agreement
does not by itself demonstrate a harmful decision change. Conversely, this
single small check cannot certify safety. The per-stage dose caveat remains.

Dense A→future attention mass is highly stable across these inputs (rank
correlation **0.970**) but correlates only **0.173** with measured action
sensitivity. Future V→V mass correlates **0.486** with action sensitivity.
These are descriptive associations, not causal evidence for the AV–VV bridge;
attention magnitude cannot substitute for the intervention measurement.

## Additional causal control and predeclared combined comparison

The real-checkpoint causal control at **`42cd75b`** completed **03:13:39.597–
03:15:10.120 UTC**, exit 0. Each of 24 heads on each of three inputs received
the same strong VV restriction, but only in the last layer at sampling step 9.
All **72/72 final normalized and denormalized actions were bitwise unchanged**,
while future-video relative L2 ranged from 0.00329 to 0.07473. This is a real
video perturbation with no remaining causal route to the final action, and
supports the probe's isolation of action rows. It does not establish that
earlier video perturbations are safe. Conditioned frames and end-of-input
Dense controls also passed. The recorded admission on GPU 0 showed existing
lightly utilized memory occupancy; this was a shared-host correctness check,
not a latency benchmark. See the [complete control report](evidence/head-stage-20260920/causal-control-report.json).

```bash
# Pinned DreamWAM-head-control-42cd75b worktree; same environment and inputs.
PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  scripts/sparse/check_head_stage_causal_control.py \
  --inputs "$ARTIFACT_ROOT/inputs-gpu4/manifest.json" \
  --out-dir "$ARTIFACT_ROOT/causal-control-42cd75b"
```

Before seeing the complete classification, `1adeeff` fixes a combined test
at **50% future-key budget**. Every sparse alternative retains exactly 168
future blocks in every layer/stage, or **5/7 of native VV pairs** including
the unchanged structural support. Uniform gives each head seven of fourteen
future blocks. Head-only protects the twelve most action-sensitive heads per
layer (mean across stages); Head × Stage protects twelve independently in
each layer/stage. Protected heads retain all fourteen future blocks, others
retain only conditioned keys. Ties use ascending head index. Three allocation
random seeds (0/1/2, independent of the unchanged model RNG) give additional
matched-cost controls, all retained in the result.

Only the two calibration inputs select the budgets; the check-input scores
are ignored. The same evenly spaced future-key rule, full fused Dense backend
and sampler are used. Full-budget/native and per-profile repeat controls are
required. This is a combined action/video **proxy** comparison: it tests
nonlinear interaction and allocation at a fixed budget, without claiming an
optimized operator, additional speed or SR. The complete classification is
required before the experiment runs. Tests at `1adeeff` passed **13/13** on
the server, exit 0, including equal mask-permitted pair counts, unchanged action
rows under real attention and no use of check-input scores in allocations.

```bash
# EXECUTED after classification/audit, GPU 4; GPU 1 deliberately left unused.
# Pinned DreamWAM-head-allocation-1adeeff worktree, freshly admitted GPU.
PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  scripts/sparse/compare_head_stage_allocations.py \
  --inputs "$ARTIFACT_ROOT/inputs-gpu4/manifest.json" \
  --classification "$ARTIFACT_ROOT/analysis-f489335/classification.json" \
  --out-dir "$ARTIFACT_ROOT/combined-1adeeff"
```

This run completed **03:43:35.432–03:44:38.537 UTC**, exit 0, on initially
empty GPU 4, with GPU 1 deliberately left unused at admission. It measured
all seven profiles on all three inputs (21 records), repeated each profile
bitwise and included native/start/end Dense controls: **51 complete requests**
including controls. All three Dense actions also matched the original full
sweep's saved references bitwise. The recorded profiles were independently
checked against calibration-only rankings and equal mask cardinalities.

| Allocation | Calibration mean action L2 | Check action L2 | Check video L2 | Check executed-prefix action L2 |
|---|---:|---:|---:|---:|
| Dense | 0 | 0 | 0 | 0 |
| Uniform | 0.164616 | 0.153968 | 0.478101 | 0.088408 |
| Head-only | 0.095173 | **0.082719** | 0.400961 | **0.042672** |
| Head × Stage | **0.090754** | 0.084239 | **0.379731** | 0.046430 |
| Random allocation seed 0 | 0.411782 | 0.293426 | 0.743059 | 0.248743 |
| Random allocation seed 1 | 0.421112 | 0.242676 | 0.725071 | 0.245445 |
| Random allocation seed 2 | 0.363392 | 0.215635 | 0.714801 | 0.193963 |

Lower is better. The first two action columns use the **full normalized
32-step sampler output**, as fixed for classification. The final column
recomputes relative L2 on the **first ten denormalized controller actions**,
which the protocol actually executes. These different spaces must not be
pooled or interpreted as SR percentage-point changes. The prefix also favors
head-only over Head × Stage on the check input. All 21 records have zero
gripper sign flips; that alone does not establish task success. Translation,
rotation and maximum absolute prefix deviations are retained in the audit.

The favorable uniform comparison is real on this limited replay set. Head ×
Stage also reduces video change more than head-only, while its check-input
action change is slightly larger. No full benchmark SR or optimized
full-request latency was measured for these new profiles. **Mask cardinality
is not executed FLOPs**: these diagnostic masks still run through the full
fused Dense operator. The current roughly 1.78× / 1.81× cache timings remain
separate evidence for different policies.

Next: expand calibration beyond these initial scenes and intervention dose,
retain head-only as the action baseline, and measure the incremental AV/VV
route and actual sparse execution at matched budgets. Any selected combination
still requires full paired SR; the user decides tolerance after complete
Pareto results. The extra Stage factor is not accepted on the strength of
this one favorable comparison against uniform allocation.
