# DreamWAM paper-evaluation handoff

Active isolated branch: [DIDO-guided action/video profile](../implementation/dido-sparse-profile/PROGRESS.md).
The [native profile and bounded diagnostics](../implementation/dido-sparse-profile/RESULTS.md)
are verified: nine observations, 18 native capture calls and 47 interventions
plus two controls, with raw arrays and parity/replay checks. The subsequent
[400-call online screen](../implementation/dido-sparse-profile/ONLINE-RESULTS.md)
and four CUDA graph cases pass. The [380-call refresh/budget/structure follow-up](../implementation/dido-sparse-profile/FOLLOWUP-RESULTS.md)
and CPU OSMesa preflight also pass. Final retained-choice timing, adapter and closed-loop
gates remain open; episode attempts are 0/50. The user
sets a 5-percentage-point SR margin and a cap of 50 **total** episode attempts
across arms for this effort, superseding the historical per-arm scope below.
See the [raw-data distinction](../implementation/dido-sparse-profile/SOURCES.md)
and [current execution record](../implementation/dido-sparse-profile/EXECUTION-20260920.json).
The [online selectors/pooling implementation](../implementation/dido-sparse-profile/ONLINE.md)
passes CPU/CUDA and real-checkpoint D/R checks;
[failures, corrections, admission refusal and all completed stages](../implementation/dido-sparse-profile/ONLINE-VERIFICATION.json)
are preserved. Shared context and layerwise value-aware routing remain development
choices; neither action-vector similarity nor the roughly 2× warm speed proves SR.

Latest [routing/profile expansion and lower-budget pilots](hybrid-routing-results-20260920.md): 57 configuration/dataset cases, 999 audited timings and 153 dependency interventions completed. The measured no-refresh uniform-anchor feature-cache candidate reaches **3/3 vs Dense 3/3 at nominal KV18.75 (56/294 rows)**, with **2.077× warm inference**; KV25 also succeeds 3/3 at 1.988×. Neither tiny pilot certifies SR preservation, and whole-episode CPU-rendered time does not improve. [V5 method interfaces](decision-support-method-20260920.md) now separate read value, refresh urgency, feature reuse and structure-only reuse. First-layer action/context routing has not shown a benefit; structure-only reuse reaches only 1.06–1.13×. Adaptive refresh remains unimplemented, and these results must not be labelled complete M1–M3 validation. All owned GPU jobs have exited; 50-pair testing remains unstarted.

Earlier [H100 / OSMesa hybrid pilots](hybrid-pilot3-osmesa-20260920.md) are complete: tasks 0/1/2 × init 0, with independent matched Dense runs. Compact KV25 is fast (1.865× warm inference) but fails 0/3 against Dense 3/3. Keeping 75% K/V succeeds 3/3 against Dense 3/3, with 1.821× warm inference and a separate same-input 36-request recheck at 1.775×. Cold-inclusive pilot inference is only 1.418× and CPU-rendered whole-episode time does not improve. These are tiny adaptive development pilots, not SR preservation or 50-pair completion; KV75 was retained before the follow-up above. All 12 episodes reached benchmark-owned terminal outcomes, with no errors or forced cleanup.

Implementation/debug snapshot before the pilots: [explicit hybrid schedules and compact cache reuse](hybrid-sparse-reuse-20260920.md). The opt-in path separates refresh timing, query budgets and cached-key read budgets, with 46/512-candidate generation, matched query quotas, resumable paired timing and frozen-profile export. Initial verification: 305 CPU tests, 36 H100 hybrid tests, and 30 real-checkpoint debug timings with own-eager parity. The first compact candidate measured 139.28 ms versus stronger Dense 262.24 ms (1.883×), but was not faster than the legacy full-K/V cache (136.72 ms) and had large action differences. Complete schedule search and larger-cohort hybrid SR remain unmeasured. The fresh-token failures and its unstarted 50-pair comparison remain separate from this new method.

**User decision, 2026-09-20 11:31 UTC:** preserve the fresh-token results and wait for an empty card before SR testing. The matched Dense/Sparse comparison remains 50 episodes per arm; no SR rollout is launched in this update.

This page records DreamWAM Joint integration with [action-eval](https://github.com/winbeau/action-eval). **main** is the maintained evaluation branch and default. The user **resumed GPU 4–7 work** on 2026-09-20 for [every-step 10% visual tokens without cross-step visual caching](fresh-visual-tokens-20260920.md). That new method is numerically verified and measures **1.191×** (269.27 → 226.12 ms) versus strengthened Dense in a shared-load 96-request experiment. It has substantial action differences and no SR result. The shared-renderer smoke failed on its first native read (`unwritten_rgb`, exit 73), before any valid episode; all owned processes exited. The new matched 50-pair comparison awaits an available renderer. The historical pause audit and older cached results are retained in the [earlier consolidated record](sparse-results-paused-20260920.md); old 500-episode queues remain stopped.

The [fixed-budget compact Head/Stage execution trial](head-stage-execution-20260920.md)
completed 300 timings: compact Head × Stage is **338.48 ms** versus **306.95 ms**
for the same mask, a negative **0.907×** result. All evidence is preserved.
The user's renewed priority is a working sparse method, so current development
returns to the existing action-guided visual token recomputation candidate.
The [single common factor](prompt-cache-single-factor-20260920.md) of exact prompt
encoding reuse has now completed **96 balanced full-request timings on three real
inputs**: **137.37 ms Sparse versus 257.20 ms equally optimized Dense (1.872×)**.
All timed actions matched their own uncached eager reference bitwise. First
instruction misses with warm graphs measure 1.791× and are reported separately.
Both measured paths now pass real-checkpoint evaluation-adapter checks, including
episode reset and changing observations under the same instruction. Complete
candidate SR remains pending. The [new matched pilot](prompt-cache-paired-20260920.md)
uses the same dedicated renderer for both arms and completed **15/15 successes
for each**, with no task failures or errors, at 05:42 UTC. Initial image pairs
still differ bitwise between runs; this pilot does not prove SR preservation.
The user subsequently requested **50 episodes per candidate** for quick method
screening, with unified full tests deferred until method selection. The 500-episode
queue was stopped: Dense retains **129/500 successes**, Sparse was not started,
and the legacy Long recovery retains **15/500 successes**. Their manifests and
accepted result hashes remain intact; incomplete full-run SR remains null.
The [new exploratory comparison](quick50-20260920.md) covers **all 10 Spatial
tasks × initial states 0–4**, independently for matched Dense and guided Sparse.
It is now paused with **five accepted Dense successes out of 50 planned**;
Sparse has not started. These five results and their manifest are preserved.

[Offline Head × Stage calibration](head-stage-calibration-20260920.md) is **complete**: 6,480 interventions, 2,160 typed units and a matched-budget combined test, with controls/audits passed. Check-input action relative L2 improves from **0.154 uniform to 0.084 Head × Stage**, while **head-only is slightly better at 0.083**. Relative-type agreement is 63.5%; Stage's incremental action benefit is unproven. The CSV, heatmaps, stability plots and all combined actions are available in the record. These masks were exercised in full offline inference; the fast SR policies remain frozen and do not load them. No new M1 speed or SR claim is made.

[Paper-mechanism audit](paper-mechanism-status-20260920.md): the current fast paths do **not** activate the paper's Head×Stage classification or complete AV–VV context route. Their measured acceleration comes from visual temporal reuse and transformer graphs; the guided path adds AV-conditioned drift selection. These results cannot be labelled as end-to-end validation of M1–M3. The new [interval-10 single-factor control](visual-cadence-single-factor-20260920.md) measures **1.973×** (265.09 → 134.36 ms) against the same strengthened Dense, but has **no official SR** and no action-guided selection. Both full timing sets, including a shared-load 1.922× result, are preserved.

Audited legacy coverage at **2026-09-20 05:24 UTC**, with Long updated after the user-requested stop at **06:15 UTC**: [matched Dense Spatial remains complete at 492/500 successes (98.4% SR)](temporal-suite-expansion-20260920.md#matched-dense-spatial-complete-long-moves-to-an-independent-gpu-3-run), compared with 493/500 in original Dense. Their 500-episode pairing has six regressions and five gains, Δ −0.2 pp, paired bootstrap 95% interval [−1.4, +1.0] pp; baseline repeatability is not established. Eager temporal has **455/500 outcomes (454 successes, one failure)** and graph temporal **428/500 (426 successes, two failures)**; both workers are inactive. Eager recovery batch 10 ended when its resource window closed and carries one zero-outcome invocation. Graph recovery batches 09/10 made no attempts, so the two zero-outcome invocations from batch 08 still leave only one further invocation in that stagnation budget. Independent Long GPU 3 stopped with **15/500 outcomes, all successes**. All earlier accepted result hashes are preserved. Object / Goal / GPU 6 Long remain stopped at their stagnation caps. Guided graph Spatial remains at 0/500; older guided outcomes stay separate. No complete candidate benchmark SR is available. Full four-suite testing is deferred until method selection under the user's new 50-episode screening plan. The user will decide SR tolerance after reviewing the Pareto results; no tolerance or non-inferiority conclusion is assumed.

Pre-prompt performance control: [conditioned-frame reuse](conditioned-frame-single-factor-20260920.md) exploits 98 empirically invariant video tokens and improves both sides. With this common optimization, complete requests measure **142.87 ms temporal versus 254.80 ms Dense (1.783×)**; unchanged guided 10% measures **140.49 ms (1.814×)**. The earlier approximately 1.97× / 2.05× results used Dense without that optimization and do not establish 2× against the stronger control. All 492 timed requests in those factor/control measurements passed own-eager parity; conditioned-frame variants also matched their pre-factor actions. Those live SR runs retain their frozen policies; the newly verified adapter exposes conditioned-frame reuse for the new matched control. The [eager guided candidate](visual-token-cache-single-factor-20260919.md) retains 95/500 outcomes including two failures where Dense and temporal succeeded. Small pilots and tensor parity do not establish benchmark quality. [FFN/neuron negative results](ffn-cache-single-factor-20260919.md) remain preserved. The user permits idle or lightly occupied GPUs; historical pause and GPU-7-only notes no longer govern this work. Each new launch rechecks resources, leaves another available card unused by this task, and preserves the other evaluation's dedicated rendering cards.

The first-input audit also found different camera-image hashes before any policy-dependent action, including 10 same-Dense identities on the same GPU with identical state vectors. Hashes do not quantify image error, but repeatable reference observations have not been established. The successful pilot counts therefore do not isolate a decision-preservation effect or prove non-inferiority. See the [input audit and current recovery state](visual-token-cache-single-factor-20260919.md#first-input-repeatability-is-not-established).

The [K/V copy-schedule factor](visual-kv-staging-single-factor-20260920.md) passed 1,008 timed eager-parity checks, but its balanced confirmation found only **0.86% additional temporal speedup** under shared load. The subsequent [DiT pre/post graph factor](dit-boundary-graphs-single-factor-20260920.md) passed 768 checks and produced **no useful net gain**. Both remain experimental and leave the active SR policies unchanged. Earlier timestamped progress snapshots remain historical records.

## Provenance and verification

Upstream: hustvl/DreamWAM, reference commit `7c35d7d094b86fc65721cd26dfbc3194addb8fd0`. Earlier server pilot: Spatial tasks 0/1/2 × 10 initial states, 30/30 success; six native aborts required recovery. This is not four-suite evidence. See action-eval `docs/verification/P3-DREAMWAM-PILOT-COMPLETE.md`.

The adapter loads the injected checkpoint, reports executed options, preserves preprocessing ownership and refuses silent CPU fallback. Graph policy integration at **2c02c5c** passed **206 sparse tests** on the server, including actual CUDA graph execution through the adapter. Native-equivalent matched controls pass bitwise action comparisons. These checks establish implementation behavior; only complete official episode pairing can establish measured benchmark quality. The separate baseline-suite effort is tracked in the [action-eval handoff](https://github.com/winbeau/action-eval/blob/main/docs/plan/00-current-handoff.md).

## Remaining work

Current follow-up prioritizes broader validation of the low-budget hybrid
candidate and a better causally calibrated routing signal. The bounded pilots
above are complete; historical fresh-token/50-pair/full-suite queues below are
not implicitly resumed.

1. Preserve the server dependency inventory; reference Python 3.10.20 and torch 2.7.1+cu126. Current pyproject delegates to requirements.txt; preserve both where possible. Recover/generate a complete lock only on server and record justified differences, including build-system pins.
2. Complete the new fresh-token matched 50-episode exploratory comparison after its renderer admission check, and use quick cohorts to select the method. Resume unified full testing only in the later phase requested by the user. Preserve completed Dense short suites and all historical partial manifests; their records cannot silently fill a new denominator.
3. Evaluate action guidance with its own matched budget/cadence control and the strengthened Dense control. Its small speed advantage over the strengthened temporal candidate does not establish a quality benefit.
4. Preserve all attempts, impose finite recovery limits and report complete coverage before SR. Native EGL aborts remain unresolved; do not assume resume makes them statistically harmless or apply unverified rendering workarounds.
5. Keep further optimizations separate, with complete `predict_action` timing and their own quality evidence. Any extension to another suite must use its official protocol; do not reuse Spatial's horizon blindly.

## Records and acceptance

Use PLANNED / VERIFIED / FAILED / BLOCKED statuses. Record timestamp/timezone, code and checkpoint hashes, environment lineage, physical GPU UUID, exact command, exit code, raw artifact location, observations and remaining limitations. Never turn an unexecuted command into a success claim.

The central [implementation checklist](https://github.com/winbeau/action-eval/blob/main/docs/plan/07-paper-baselines-checklist.md) tracks baseline-suite integration. The linked factor records above contain the commands and artifacts actually executed for this acceleration goal. No dependency installation was performed during these factor measurements.
