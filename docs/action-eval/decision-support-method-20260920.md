# Decision-support routing: executable hypotheses for Introduction V5

Status: routing, structure-only reuse and offline interventions implemented;
exploratory screening completed, 2026-09-20 UTC. First-layer decision/context
ranking has **not** demonstrated a benefit. See the complete
[57-case table](evidence/hybrid-routing-20260920/results.csv), including negative results.
The Introduction's three findings are **hypotheses to test**, not established
facts. In particular, small action error is not sufficient control quality,
attention mass is not causal importance, and latent fidelity is not video quality.

## What is selected, and why?

For each visual token j, let d[j] be its first-layer A→V probability mass,
averaged over heads and summed over action queries. Softmax includes **both
visual and action keys**, with the original visibility mask. DreamWAM Joint
allows action queries to read **all visual frames**. Video queries in the
observed first frame cannot read future frames; future video queries can read
all video frames. Video queries never read action keys.

Online profiling uses only current action/visual inputs. It never sees future
denoising steps, Dense teacher actions, task outcomes or privileged simulator
states. The new selectors explicitly pay for one current full-video first-layer
K projection. This is a separate experimental factor from legacy action-drift,
which probes cached K and uses one score for both refresh and retention.

Decision-support context is a **backward dependency expansion**. Take the top
ceil(N × support_seed_ratio) seeds by d (default 0.1), project only those visual
queries, and compute their native V→V probabilities P. Then

```
s[j] = sum(seed i) normalized(d[i]) * mean_heads(P[i,j])
read_value[j] = mean_normalize(d)[j] + context_weight * mean_normalize(s)[j]
refresh_urgency[j] = token_drift(current[j], last_computed[j])
                     * (1 + guidance_weight * mean_normalize(read_value)[j])
```

This uses Pᵀd, not Pd: select the **keys supporting important query states**,
not the queries consuming important keys. Original RoPE positions and masks
are retained. One hop and layer 0 are explicitly a cheap proxy, not a claim to
have recovered complete multilayer spatial/temporal/physical causality.

At Dense step 0, choose the top read_value tokens within fixed balanced
per-frame quotas. At a feature-refresh Sparse step, choose queries U by
refresh_urgency; retain the best read_value members of the old route to fill R.
Each new read key must be freshly computed: R_new − R_old ⊆ U ⊆ R_new.
The global budget is never enlarged for seeds or frame coverage. Queries and
retained keys have different purposes: an important unchanged key should remain
readable even when its refresh urgency is zero.

The frame quotas, support seed budget, context weight, guidance weight, Q/KV
budgets, backend and schedule are explicit fingerprinted options. A future
nonuniform frame-allocation study must be a separate ablation, not a silent
preserve-observed-frame exception.

## Independent selector ablations

| Selector | Read ranking | Refresh ranking | Current full K probe |
|---|---|---|---|
| uniform | uniform positions / old deterministic fill | uniform | no |
| drift | uniform at anchor, cached-input drift later | drift | no |
| action_drift | legacy cached-K A→V at anchor, drift-weighted later | same legacy score | no |
| visual_context | incoming V→V mass, uniform weights over all visual query seeds | drift × visual importance | yes |
| action | current A→V mass | drift × action importance | yes |
| action_context | current A→V + backward V→V support | drift × support importance | yes |

`action_context` with context_weight=0 is exactly `action`; tests require identical
routes and actions. Thus that comparison isolates context expansion. Comparing
new `action` with old `action_drift` changes both key freshness and read scoring;
it is not a single-factor attribution. `visual_context` is a generation-side
attention proxy, not a validated generation-criticality oracle.

## Two genuinely different reuse mechanisms

`reuse.mode: features` is the existing fast method. Dense anchors cache full
per-layer K/V and final hidden state. Sparse steps update U; Reuse steps update
only actions and read packed old visual K/V. Unselected visual hidden rows remain
cached. This is **feature reuse**, not just reuse of a sparse pattern.

`reuse.mode: structure` matches V5's stated structural mechanism. Dense step 0
creates a route; Sparse steps rebuild it; Reuse steps keep the indices but
recompute **every retained visual Q/K/V, attention, FFN and world residual**.
No old visual K/V is supplied to its transformer executor. Unselected rows
bypass from the **current** pre-transformer input. Hence Q ratio must equal KV
ratio, compact read is mandatory, and selectors must use current inputs.
This distinction is tested by poisoning old features and by eager/CUDA-graph
parity on changed observations. Structure-only speed need not match feature-cache
speed: that is an experiment, not an assumption.

Both modes retain all action rows, ten denoising steps, full-grid prediction
heads, both samplers, the original weights and the evaluator protocol.
Canonical feature caching is not claimed to compress KV storage.

## Experiments that can support or falsify V5

1. **Budget curve:** six KV budgets with the original selector, fixed Q=10%,
   Dense=0, Sparse=5, Reuse elsewhere. Same-input strengthened Dense timing.
2. **Selection:** six selectors × KV={12.5%,25%,37.5%,50%}, same Q/schedule,
   all route/probe/packing costs inside complete request timings. Separate
   untimed traces retain actual positions, cache ages and phase spans.
3. **Dependency interventions:** on native dense trajectories, remove the same
   balanced 10% of visual keys at steps 1/5/9, across all layers. Compare uniform,
   top A→V, bottom A→V, visual-context and action-context removals. Record final
   action error (including the executed ten-action prefix), gripper disagreement
   and **future latent distance to Dense**, excluding the observed frame. This
   jointly perturbs AV/VV read support and is not equivalent to deleting token
   computation. It cannot establish full causal mechanisms by itself.
4. **Cross-step stability:** retain complete first-layer score vectors and
   fixed-budget supports along the untouched Dense trajectory. Report adjacent
   Jaccard, by method and step. Uniform observed-frame behavior cannot be used
   as evidence that future support is stable. Add per-frame analysis in reporting.
5. **Refresh position:** enumerate no refresh and each single refresh in steps
   1–9 at a fixed budget/selector. Multi-refresh candidates use the same manifest
   interface. Step 5 is a baseline, not a claimed optimum.
6. **Reuse semantics:** run structure-only at equal Q/KV budget beside feature
   reuse, not under a shared ambiguous “reuse” label.
7. **Closed loop:** test shortlisted low-budget candidates against matched
   strengthened Dense on a finite pilot. Success comes only from LIBERO.

Adaptive refresh is **not implemented or validated yet**. A threshold must be
calibrated on a development trajectory set using measured route drift and its
action sensitivity, then frozen before confirmation. A fixed Sparse=5 schedule
must not be described as adaptive. The current explicit/periodic/hash-frozen
profile interfaces already permit measured refresh-step selection without
embedding search in the policy. The three historical observations are exposed
debug inputs, not held-out confirmation or representative trajectory coverage.

The offline audit also supports `--scope future`: observed-frame keys are never
removed. `--group-count 7` adds a partition of each future frame into seven
within-frame index groups, measuring action/video sensitivity independently of
attention ranking. All groups and steps are retained, not only favorable ones.
These effects concern executable denormalized/binarized actions and future latent
distance to Dense, not true world-prediction quality or closed-loop task success.

The existing [Head × Stage sensitivity study](head-stage-calibration-20260920.md)
is a different intervention granularity. It is not silently integrated into the
new token selector. Its negative compact execution results and the new first-layer
negative findings remain separate evidence. Neither speed nor scientific claims
can be inherited just because both components are called decision-aware.

For trajectory coverage, action-eval's existing `outputs.save_observations` option
now records actual policy inputs without changing actions or success judgement.
`export_trajectory_inputs.py` validates full terminal coverage (including failures),
checks archive and policy-input hashes and selects first/middle/last calls per
episode. It exports only images/state/instruction, no outcomes or teacher actions.
This round captures nine snapshots from three Dense development episodes. They
are correlated within episodes and explicitly labelled development.

## Entry points and implementation boundaries

- `dreamwam/sparse/hybrid/routing.py`: causal read/refresh scores and backward support.
- `selection.py`: fixed-budget sets and valid feature-cache swaps, no scheduling.
- `execution.py`: actual dense/sparse/fresh/reuse tensor computation, no selection.
- `state.py`: request-local feature state versus current-input bypass semantics.
- `runtime.py`: orchestration and explicit executed-work/probe diagnostics.
- `profiling.py`: offline-only intervention hooks; never imported by online policy.
- `generate_hybrid_schedules.py`: finite Cartesian budget/selector/step manifest;
  rejects duplicates, invalid budgets and over-budget grids without silent truncation.
- `benchmark_hybrid_schedules.py`: complete-request timings, own-eager checks,
  strong Dense pairing and immutable resume provenance.
- `profile_hybrid_dependencies.py`: dense trajectory/intervention artifacts, no speed/SR claim.

Runtime configuration remains opt-in. Default historical policy hashes are
preserved when new factors are unused. Development follows local edit/commit/push
then clean server pull, frozen run worktrees and unchanged environments.
