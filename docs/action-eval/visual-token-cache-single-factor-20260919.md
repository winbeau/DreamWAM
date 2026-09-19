# Joint visual token refresh and current-action guidance

Status: **MEASURED latency; official quality validation in progress**.
Date: 2026-09-19 UTC. The SR-constrained acceleration goal remains unproven.

Update at **23:36:39 UTC**: the full guided run has **95/500** terminal outcomes
(93 successes, two official task failures). Dense and the temporal-only control
succeeded on both failed identities. Guided is now inactive after a bounded
recovery batch; temporal-only full evaluation has resumed from 118 outcomes.
[Separate graph factors](visual-graph-dispatch-single-factor-20260919.md) reach
2.064× against equally graphed Dense with bitwise eager parity, but do not repair
these task failures or establish SR preservation. See that record for the
accepted result hashes, new graph pilot and current allocation.

These are two successive factors on the [interval-5 temporal cache](visual-cache-single-factor-20260919.md).
V3 changes only the number of visual tokens recomputed at the second refresh.
V4 adds current-action guidance to that selection, holding cadence and budget fixed.
Both retain all ten action denoising steps, both original schedulers, full visual
key columns, bf16, horizon 32, replan 10, 9 video frames, seed 42 and the original
separate CPU RNG generators. Neither enables a neuron cache or public operator changes.
The unchanged Joint checkpoint SHA-256 is
`6c087e5b9e201f19dbe92834b470a7ee18937a18ea9adcbb49886aa1e3ac4c61`.

## V3: physically refresh selected visual tokens

[`VisualTokenRefreshCache`](../../dreamwam/sparse/visual_token_cache.py), commit
**062e838**, computes a dense anchor at step 0. At step 5, it ranks accumulated
transformer-input drift and recomputes 30 of 294 visual tokens (ceil of 10%).
Queries, projections, FFNs, cross-attention and world residuals operate on these
selected rows at all 30 layers. Each layer replaces their cached K/V rows while
retaining all original key columns and the original visibility mask. Final visual
rows are scattered into the full grid for the unchanged video prediction/scheduler.
Unselected input references advance only when their rows are recomputed.

Steps 1–4 and 6–9 use the temporal cache. A request computes 9,720 of 88,200 visual
token-layer rows, versus 17,640 for temporal-only interval 5. This is **11.02%** of
Dense visual token-layer work, including the mandatory dense anchor; it is not a
claim of 10% total model compute. All 300 action layer updates remain.

H200 NVL GPU 2 (`GPU-b15ccd2e-b17d-20a9-a130-b4b7eb37a074`), ending
**21:55:09 UTC**, exit 0. Two warmups and 24 complete requests per variant, rotating
order over the same three synthetic inputs. GPU 2 had no other CUDA process in
the recorded before/after snapshots. Other cards were in use.

| V3 variant | Mean ms | p50 ms | p95 ms |
|---|---:|---:|---:|
| Native Dense | 461.87 | 461.33 | 464.39 |
| Matched Dense | 465.46 | 465.24 | 468.18 |
| Temporal-only, interval 5 | 282.53 | 282.37 | 284.36 |
| Drift-selected 10% token refresh | 282.06 | 281.97 | 284.17 |

The candidate is **1.650×** matched Dense, but only **1.00167×** temporal-only.
The additional token compression has no substantial measured speed benefit in
this implementation; the overall gain comes from temporal reuse. Maximum synthetic
action relative L2 is 0.12654, a diagnostic with no SR interpretation.

Full-budget actions are bitwise native Dense; the all-token interval-5 control is
bitwise the earlier temporal cache. Parameter versions are unchanged. The 36-test
cache suite passed, including actual query lengths, complete key masks, preserved
unselected K/V/output rows, scheduler counts and exception cleanup.

Evidence: [manifest](evidence/visual-token-cache-20260919/unguided/manifest.json),
[96 requests](evidence/visual-token-cache-20260919/unguided/requests.jsonl),
[summary](evidence/visual-token-cache-20260919/unguided/summary.json),
[telemetry](evidence/visual-token-cache-20260919/unguided/keep010-gpu.csv),
[tests](evidence/visual-token-cache-20260919/unguided/tests.log).

## V4: add current-action guidance

[`ActionGuidedVisualTokenCache`](../../dreamwam/sparse/action_guided_visual_token_cache.py),
commit **4ffa7ae**, uses first-layer Q/K from the **current** action state at the
partial refresh. The original jointly normalized A→[V,A] attention over cached
visual keys supplies visual relevance. The selection score is input drift times
`1 + guidance_weight * relevance / mean(relevance)`, with weight 1 in this run.
The selected mask is shared across layers. Zero input drift stays zero regardless
of relevance. No future denoising step, next request or task outcome enters selection.

This adds one action Q/K/V projection and one attention-mass calculation per
request, both counted and timed. It does not add an action FFN or scheduler update.
Weight 0 is bitwise the unguided selector; the full-budget control is bitwise Dense.

GPU 2, ending **22:02:21 UTC**, exit 0. Two warmups and 30 complete requests per
variant, 150 measured requests total. This factor uses its own contemporaneous
Dense, temporal-only and unguided controls; cross-run latency changes are not
attributed to guidance. GPU 2 again had no foreign process in the recorded
before/after snapshots.

| V4 variant | Mean ms | p50 ms | p95 ms |
|---|---:|---:|---:|
| Native Dense | 450.15 | 450.00 | 453.22 |
| Matched Dense | 455.81 | 455.41 | 459.42 |
| Temporal-only, interval 5 | 275.26 | 274.77 | 278.26 |
| Unguided 10% token refresh | 274.68 | 274.33 | 276.51 |
| Action-guided 10% token refresh | 275.09 | 274.38 | 277.36 |

Guided refresh is **1.657×** matched Dense (**1.636×** native Dense) and **1.00065×** temporal-only.
It is **0.99854×** unguided refresh: guidance adds 0.402 ms mean latency.
Whether guidance improves decision quality is unknown. Maximum synthetic action
relative L2 is 0.12504 and is not evidence of preserved SR. All parity and parameter
checks passed; the expanded cache suite passed 41 tests.

Evidence: [manifest](evidence/visual-token-cache-20260919/guided/manifest.json),
[150 requests](evidence/visual-token-cache-20260919/guided/requests.jsonl),
[summary](evidence/visual-token-cache-20260919/guided/summary.json),
[telemetry](evidence/visual-token-cache-20260919/guided/guided010-gpu.csv),
[tests](evidence/visual-token-cache-20260919/guided/tests.log).

## Environment and reproduction

Both timers cover synchronized **full `predict_action`**, including text/VAE
encoding and CPU action output. Inputs are three seeded synthetic camera pairs,
zero proprio and a fixed instruction; these are latency measurements, not rollouts.
The existing Python 3.10.20 / torch 2.7.1+cu126 environment was reused without any
install or sync. Model `uv.lock` remains absent; source/dependency-file hashes and
the actual environment are recorded in each manifest.

Fixed server worktrees are `dreamwam-sr/DreamWAM-token-062e838` and
`dreamwam-sr/DreamWAM-guided-token-4ffa7ae`. Run in the corresponding worktree:

```bash
PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=2 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python scripts/sparse/benchmark_ffn_context_cache.py \
  --cache-kind visual_tokens --keep-ratio 0.1 --refresh-every 5 \
  --warmup 2 --reps 24 --out-dir outputs/token-refresh-20260919/keep010

PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=2 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python scripts/sparse/benchmark_ffn_context_cache.py \
  --cache-kind guided_visual_tokens --keep-ratio 0.1 --refresh-every 5 \
  --guidance-weight 1 --warmup 2 --reps 30 \
  --out-dir outputs/guided-token-20260919/guided010
```

## Policy integration and official pilot

Model commits **f44df70/8390f6a** expose strictly validated model-owned options:

```yaml
visual_cache:
  refresh_every: 5
  token_keep_ratio: 0.1
  action_guidance_weight: 1.0
```

The legacy interval-only option keeps its behavior and fingerprint. Token selection
without an explicit guidance weight defaults to zero guidance. Unknown keys,
non-finite values and invalid bounds fail before loading weights. Adapter code
passes through options and publishes actual model configuration and execution
counts; it makes no selection or success decisions. **187 sparse tests passed**
on the server, including real Joint sampling through the adapter. The first test
invocation failed collection on a test-helper import, corrected in 8390f6a; its
failed log is retained under the initial preflight directory.

The new pilot config is committed in action-eval **13f4a2a**. Runtime evaluator
remains frozen at **4701ac2**; model is fixed at **8390f6a**. Validation and doctor
passed before launch. Pilot started **22:20:45 UTC** on GPU 2, with the same 15
task/init/repeat/seed identities and `dreamwam-release-v1` protocol as the completed
Dense pilot. It does not replace the frozen V1 500-episode candidate. GPU 2 was
lightly occupied by an independent diagnostic at launch (0%, 24,743 MiB), so this
rollout is explicitly shared and supplies no isolated latency claim. No foreign
process was signalled, and we did not allocate GPU 3.

Server artifacts: `dreamwam-sr/outputs/guided-visual-token-pilot-20260919/`, run
`guided/run-20260919T222045Z-aed6d97d`; validation/tests are under
`preflight-8390f6a/`. At **22:21:54 UTC**, the actual worker fingerprint confirmed
the checkpoint hash, guidance/budget/cadence and preserved scientific settings;
2/15 terminal outcomes were present, both successes. The worker was alive then.
This is an incomplete pilot; no SR or guidance-quality conclusion is made.
The next gate is complete official pairing, followed by full Spatial coverage.

### Native recovery and resource allocation

The GPU-2 pilot stopped with **3/15 accepted successes, 0 task failures**. Its
initial process and first recovery both aborted natively. At **22:29:33 UTC**,
the recovery supervisor observed 75% GPU utilization after its own worker exited
and stopped with `gpu_resource_window_closed`. The [partial bundle](evidence/visual-token-cache-20260919/gpu2-partial/summary.json)
keeps `success_rate: null`; [recovery state](evidence/visual-token-cache-20260919/gpu2-partial/recovery-summary.json)
and [resource checks](evidence/visual-token-cache-20260919/gpu2-partial/gpu-checks.jsonl)
are retained. Abrupt native exits did not persist a per-episode error record, so
the remaining bundle rows say `not_run`; the native errors remain in process logs.
These rows must not be described as tasks that failed, or as proof no invocation
attempted them.

A separate GPU-6 pilot began **22:32:10 UTC**, run
`guided-gpu6/run-20260919T223210Z-9d7852f8`, with identical model and scientific
settings. It reached **5/15 accepted successes**. The initial process and five
recoveries ended in SIGABRT; the last three recoveries added no outcome, so the
supervisor stopped at `stagnation_limit`. No outputs were pooled across GPUs.

The known productive GPU-1 allocation was then assigned to the action-guided
candidate. The older temporal 500-episode run was requested to stop via its own
supervisor at **22:50:46 UTC**. Its frozen config/manifest and **118 settled
successes** were checked unchanged after exit; its policy worker was confirmed
gone. That temporal run remains available for later resume and has no complete SR.

The new GPU-1 pilot began **22:51:59 UTC**, run
`guided-gpu1/run-20260919T225159Z-9084dbca`, after validation/doctor and a free-card
snapshot. The initial process completed all five task-0 episodes, then exited 134.
Bounded recovery started **22:54:25 UTC**, with ten invocations maximum and three
consecutive no-progress invocations as the stopping limit. The original GPU-2
and GPU-6 manifests remain separate. The full 500-episode guided config, committed
in action-eval **d632135**, passed validation/doctor before launch.

By **22:58:36 UTC**, the GPU-1 pilot had completed **15 accepted successes, 0
task failures**; recovery 3 returned 0. The [complete paired pilot](evidence/visual-token-cache-20260919/gpu1-pilot/paired-pilot.json)
contains all 15 planned pairs, each successful on both sides. The same first-input
audit still finds differing image hashes on every pair, with identical state
vectors. This is a successful plumbing pilot with the repeatability limitation
below, not evidence of non-inferiority. [Episode records](evidence/visual-token-cache-20260919/gpu1-pilot/per_episode.csv),
[executed provenance](evidence/visual-token-cache-20260919/gpu1-pilot/provenance.json)
and [recovery completion](evidence/visual-token-cache-20260919/gpu1-pilot/recovery-summary.json)
are retained.

The full guided Spatial run began **22:59:43 UTC** on GPU 1, using model
**8390f6a**, config **d632135** and evaluator **4701ac2**:
`dreamwam-sr/outputs/guided-visual-token-spatial-20260919/guided-gpu1/run-20260919T225943Z-b4cfcb70`.
It uses all 10 tasks × 50 initial states, repeat 0, seed 42 and the unchanged
`dreamwam-release-v1` protocol. The complete pilot was not imported into it.
The actual worker description was verified against the checkpoint hash,
guidance/cadence/budget and all scientific settings.

At **23:02:05 UTC**, guided coverage was **6/500**, matched Dense **126/500**,
and the inactive temporal run retained **118/500**, with no task failures among
these recorded terminal outcomes. The guided initial process had exited 134;
bounded recovery started **23:02:38 UTC**. At **23:03:32 UTC**, guided supervisor
148285 and Dense supervisor 142854 were verified alive. These are timestamped
handles, not continuing liveness guarantees. Full SR remains null.
Evidence: [executed coverage snapshot](evidence/visual-token-cache-20260919/full-spatial/progress-230205Z.json),
[validation](evidence/visual-token-cache-20260919/full-spatial/validate.log),
[doctor](evidence/visual-token-cache-20260919/full-spatial/doctor.log).

### First-input repeatability is not established

[`audit_initial_inputs.py`](../../scripts/sparse/audit_initial_inputs.py), commit
**8e6ee98**, compares the first saved policy input after the prescribed 30 wait
steps, before a policy-dependent action. It checks benchmark/protocol identity
and never computes SR or substitutes an intersection for an SR denominator.
Two server invocations exited 0 at **22:48:59 UTC**:

| Initial-input comparison | Shared identities | Equal state hashes | Equal agentview hashes | Equal wrist hashes |
|---|---:|---:|---:|---:|
| Completed temporal pilot vs Dense pilot | 15 | 15 | 0 | 0 |
| Dense pilot vs full Dense, both on GPU 0 | 10 | 10 | 0 | 0 |
| Completed guided GPU-1 pilot vs Dense pilot | 15 | 15 | 0 | 0 |

The second comparison keeps the same model **28845c6**, interval-1 control,
physical GPU and scientific settings. Thus image non-identity also occurs without
changing the policy method. **Hashes alone do not quantify image error or identify
its cause.** These audits do not establish that images are corrupted, but they
show that byte-identical initial observations cannot currently be assumed.
The completed 15-pair pilot cannot isolate a decision-preservation effect from
this observation variability, and cannot support a non-inferiority claim.
Official terminal records remain intact. The independent EGL investigation has
not yet established a production repair or repeatable reference observations.

Evidence: [temporal/Dense audit](evidence/visual-token-cache-20260919/input-audit/temporal-initial-input-audit-8e6ee98.json),
[same-Dense audit](evidence/visual-token-cache-20260919/input-audit/dense-initial-input-audit-8e6ee98.json),
[guided/Dense audit](evidence/visual-token-cache-20260919/input-audit/guided-gpu1-initial-input-audit-8e6ee98.json)
(the last audit ran at **22:59:41 UTC**).

```bash
.venv/bin/python scripts/sparse/audit_initial_inputs.py \
  --reference "$DENSE_PILOT" --candidate "$TEMPORAL_PILOT" --out "$NEW_AUDIT_JSON"
.venv/bin/python scripts/sparse/audit_initial_inputs.py \
  --reference "$DENSE_PILOT" --candidate "$FULL_DENSE_RUN" --out "$NEW_DENSE_AUDIT_JSON"
```

## Diagnostic profile: selection leaves the operator count largely unchanged

[`profile_visual_cache.py`](../../scripts/sparse/profile_visual_cache.py), measured
commit **8a03e8a**, completed **22:37:23 UTC**, exit 0, on GPU 7. The card was shared
with a lightly utilized foreign workload; no foreign process was signalled. It
loaded the same real checkpoint/environment and captured one instrumented request
per variant after two warmups. All three profiled outputs were bitwise equal to
their own uninstrumented outputs; matched Dense was also bitwise native Dense,
and parameter versions did not change.

The profile adds considerable overhead and the first trace has additional
profiler startup costs. **Its wall times are not speedup measurements.** PyTorch
also emits CUDA user annotations with the same stage names; those spans contain
stream gaps and must not be treated as kernel time or additional stage calls.
The [interpretation](evidence/visual-token-cache-20260919/profile/interpretation.json)
filters to CPU annotation rows and checks invocation counts. Commit **b8bb6ae**
applies this filtering in future profiler output; its import/CLI check passed.
Original raw summaries are retained without editing.

| Instrumented scope | Calls | CPU span ms | Attributed device work ms |
|---|---:|---:|---:|
| Temporal-only refresh | 2 | 153.62 | 53.16 |
| Guided-token refresh | 2 | 154.76 | 46.21 |
| Temporal-only action reuse | 8 | 296.87 | 65.86 |
| Guided-token action reuse | 8 | 293.23 | 65.93 |

Scope totals are nested and must not be added together. The recorded
`cudaLaunchKernel` call counts are 29,440 temporal-only and 29,509 guided;
`cuLaunchKernelEx` counts are 3,996 and 3,999 respectively. These API counts are
reported separately, not summed into a claimed number of unique kernels.
Selecting fewer rows leaves nearly the same operator/launch structure, while
all eight action reuse steps remain. This supports examining dispatch costs as
a separate next factor; it is not evidence that a particular graph/fusion change
will achieve 2×, and no such change is enabled in any SR run.

Evidence: [manifest](evidence/visual-token-cache-20260919/profile/manifest.json),
[matched profile](evidence/visual-token-cache-20260919/profile/matched_dense-profile.json),
[temporal profile](evidence/visual-token-cache-20260919/profile/temporal_only-profile.json),
[guided profile](evidence/visual-token-cache-20260919/profile/guided_tokens-profile.json).
The three complete Chrome traces (285 MiB total) remain on the server under
`dreamwam-sr/DreamWAM-profile-8a03e8a/outputs/visual-profile-20260919/profile/`.

```bash
PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=7 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python scripts/sparse/profile_visual_cache.py --warmup 2 \
  --out-dir outputs/visual-profile-20260919/profile
```
