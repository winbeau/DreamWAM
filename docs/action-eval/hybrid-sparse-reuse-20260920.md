# Explicit hybrid visual schedules and compact cache reuse

Status: **Core implementation and experiment interfaces IMPLEMENTED; CPU, CUDA
and real-checkpoint debug checks VERIFIED; complete search and SR unmeasured**.
Last evidence audit: 2026-09-20 12:44 UTC / 05:44 America/Los_Angeles (PDT).
This implements the requested Dense first step, selected Sparse refresh steps,
and full/compact cache reuse as a new opt-in method. Existing fresh-token H100
evaluation remains a separate frozen worktree and experiment.

User acceptance clarification (2026-09-20): prioritize **>1.5× against the matched
strengthened Dense control and an acceptable SR**, not extra speed over the old
cached method. The acceptable SR decrease is not yet numeric. Report paired
success counts, percentage-point differences and uncertainty before judging
quality; neither action error nor the debug speedup is final acceptance.

## Implementation and ownership

The modules live in [dreamwam/sparse/hybrid](../../dreamwam/sparse/hybrid/).

| Module | Owns |
|---|---|
| config / schedule | Strict options, explicit/periodic/profile plans, stable identities |
| selection | Causal uniform/drift/action-drift scores and fixed-size query/read sets |
| state | Request-local canonical/packed K/V, hidden output, positions and ages |
| execution | Dense, partial visual refresh and action-only reuse mathematics |
| graphs | Bounded eager/buffered/CUDA graph dispatch using the same mathematics |
| runtime | Lifecycle, sequential execution and actual-work diagnostics |
| search / experiment | Candidate enumeration, immutable request identities, coverage and paired timing |

The model policy installs exactly one runtime. The evaluation adapter only passes
options, validates the effective sampler step count, and publishes the executed
options, plan hash, operations and counters. The evaluator still owns SR.
The integration follows the repository's action-eval-adapter/config skills.

Commits pushed to main:

| Commit | Increment |
|---|---|
| 792d629 | Pure configuration, scheduling, profile checks, 46/512 candidate generation |
| dd89120 | Full-read runtime; legacy drift/guided equivalence |
| 3bd17cb | Compact K/V reuse and budgeted route replacement |
| aaee0a7 | Bounded graph backend and policy/adapter integration |
| c61cc89 | Finite/resumable benchmark, trace costs, report/profile export |
| 5dec9c3 | All-Dense manifest generation and real-policy numerical control |
| afe6982 | Matched full/compact query quotas and speedups against each control |
| dc029e3 | Distinguish same-attempt, cross-resume and untracked timing pairs |

All application edits were local and committed/pushed before server pull.
Tests use the separate server worktree
`/root/wenbiao_zhao/dreamwam-sr/DreamWAM-hybrid-dev`.
The existing Python 3.10.20 / torch 2.7.1+cu126 environment is reused without
installing or changing dependencies. Symlinks in this new worktree reference the
already verified external model assets.

## Exact semantics

The first step is Dense. Subsequent entries may be Dense, Sparse or Reuse.
Dense computes the native complete transformer and reanchors the cache.
Sparse recomputes selected visual QKV/attention/FFN/world-residual rows.
Reuse skips the visual transformer and recomputes every action layer.
The full-grid prediction head and both original schedulers execute on every step.
Unselected visual hidden rows retain their last computed values.

The query budget and read budget are independent. Full read defaults to legacy
global query selection (`frame_quota: none`) and reads every cached visual key. Compact read
physically shortens K/V; all action keys remain in the same joint softmax.
The original RoPE positions and original visibility mask determine valid edges.
Compact selection requires balanced per-frame query and read quotas. Full read
also accepts `frame_quota: balanced` so a K/V-read ablation can hold the query
selection rule fixed. Identical causal inputs produce identical query choices;
diverging model trajectories need not. Only read support must cover every frame,
so a very small query budget can leave some frames unchanged.

For compact refresh, U is a subset of R. Every newly introduced read key is in U
and is recomputed in that step. Remaining read slots come from the previous route.
Reuse reads the existing packed buffers without regathering canonical K/V at each
layer/step. Canonical full-sized K/V are retained to support reselection and
full-read controls: this version does not claim KV-storage compression.

Graph warmup/capture receives isolated staged tensors. It cannot advance cache
ages, sampler steps or committed request state. Outputs are owned copies.
Changed weights/device/dtype invalidate graphs; shape/layout differences use
separate bounded entries. Capture failure is an error, never a silent eager claim.
Request completion, exceptions, reset and close discard request visual state.

## Model option

This example is executable after deployment of the implementation. Steps 3 and 7
illustrate the interface; they are not a measured best schedule.

```yaml
hybrid_visual:
  schema_version: 1
  schedule:
    kind: explicit
    num_steps: 10
    operations: [dense, reuse, reuse, sparse, reuse, reuse, reuse, sparse, reuse, reuse]
  recompute: {keep_ratio: 0.10}
  read: {mode: compact, keep_ratio: 0.25}
  selection: {method: action_drift, guidance_weight: 1.0, frame_quota: balanced}
  execution: {backend: cuda_graph, graph_warmup: 3, max_graphs: 8}
  diagnostics: {level: counters}
```

Use `read: {mode: full, keep_ratio: 1.0}` as the separate full-context control,
keeping `frame_quota: balanced` for matched-query-rule ablations. Set it to `none`
when reproducing legacy global selection. Do not silently conflate these controls.
Backends are `eager`, `buffered`, `cuda_graph`; selectors are `uniform`,
`drift`, `action_drift`. Periodic schedules accept `refresh_every` and
`refresh_operation: dense|sparse` and compile into the same explicit plan.
Unknown keys, invalid budgets, first-step reuse, insufficient frame support,
and mismatch with the actual number of sampler steps fail explicitly.
Existing `visual_cache`, `fresh_visual_tokens` and VV `sparse` cannot be
stacked with this wrapper. Full-budget controls keep native numerical behavior.

## Searching steps without guessing

The generator needs only the Python standard library. Run on the server from
the frozen checkout; output directories are external to Git.

```bash
python3 scripts/sparse/generate_hybrid_schedules.py \
  --num-steps 10 --dense-steps 0 --candidate-sparse-steps 1:10 \
  --refresh-counts 0,1,2 --recompute-ratio 0.10 \
  --read-mode compact --read-ratio 0.25 --backend cuda_graph \
  --out "$OUT/schedules46.jsonl"
```

This generates 46 unique plans. `--refresh-counts 0:10` generates all 512
Sparse/Reuse combinations after the Dense anchor. Ranges are half-open.
`--frame-quota balanced` and `--guidance-weight 1` expose the corresponding
selection factors; use the same values on both full/compact manifests when
isolating the read budget. `--selection uniform|drift|action_drift` changes scoring.
For the all-Dense control, specify `--dense-steps 0:10`,
`--candidate-sparse-steps ""`, `--refresh-counts 0`,
`--read-mode full --read-ratio 1 --recompute-ratio 1`.
The generator refuses to overwrite a manifest.

```bash
CUDA_VISIBLE_DEVICES=<authorized-device> PYTHONPATH=. <verified-python> \
  scripts/sparse/benchmark_hybrid_schedules.py \
  --schedules "$OUT/schedules46.jsonl" --inputs "$INPUT_MANIFEST" \
  --split-label development --reps 6 --warmup 2 --group-size 2 \
  --out-dir "$OUT/search"
```

Each group holds a bounded number of candidates and interleaved strengthened
Dense, legacy interval-5/10%-refresh, and every-step fresh-10% controls.
All receive the same exact prompt cache and corresponding dispatch backend.
Each candidate/input/repetition has a predetermined identity and cyclic order.
Every sample times complete synchronized predict_action through CPU action output.
Model load, graph/setup warmups and first-instruction misses are separate.
Own-eager action parity and all action-layer updates are checked.

`--max-candidates` limits the declared search subset; `--max-new-requests`
stops after a finite number of new accepted samples. Resume with the same command
and `--resume`; the request budget can change, scientific identity cannot.
The benchmark rejects changed code, hardware, inputs, checkpoint, configuration,
duplicate/corrupt journal rows or changed accepted action artifacts.
Previously accepted requests are not rerun into the timing journal.
Shared GPU admission requires `--allow-shared-gpu` and remains explicitly recorded.
Each new timing row records its invocation's `attempt_id`. Reports distinguish
same-attempt, cross-attempt and older untracked pairs, and expose a separate
same-attempt speedup. Pooled pairs spanning a restart are not continuous
contemporaneous measurements; resume preserves coverage, not unchanged host load.

`--trace` adds separate instrumented hybrid requests after timing. Trace records
indices, route hash, feature/route age, actual timesteps, q/kv rows and selection,
preparation, execution and cache-commit spans. CUDA spans include stream gaps;
they are not kernel self-time. Do not sum CPU and GPU spans or nested totals.
Trace calls must preserve actions and are excluded from latency samples.

```bash
python3 scripts/sparse/summarize_hybrid_schedules.py \
  --run-dir "$OUT/search" --export-profiles
```

The report includes coverage, input/repetition-paired speedups, attempt accounting,
speedups against each requested control, action diagnostics,
and an explicitly diagnostic latency/action-error Pareto set. The shortlist includes
fastest and lowest-action-error candidates and the interval-5 point when available.
No action-error threshold certifies or rejects SR.
Incomplete comparisons have no complete paired-speedup claim; SR always remains null.

Exported `*.options.json` files are model-options fragments, not benchmark configs.
Keep the matched prompt-cache, checkpoint, protocol, horizon, precision and RNG
settings when integrating into action-eval. A profile binds a content hash,
step count, policy options, layout and scheduler; use its explicit absolute path.
The full experiment manifest retains code/checkpoint/input provenance.
Search on development episodes; frozen independent confirmation requires a declared
episode split. The three historical observations are debug inputs, not held-out data.

## Verification and remaining experiments

At source `dc029e3`, the server CPU sparse suite passed **305 tests**, with
**15 CUDA skips** and **8 subtests**, exit 0, 8.08 s. Script syntax compilation
also exited 0. The focused hybrid suite passed **36 tests and 8 subtests** on
H100 GPU 3, exit 0, 6.10 s, including actual graph capture/replay. The prior
`c61cc89` CPU run (301 passed) and `5dec9c3` CUDA run (32 passed) are retained.
Earlier CPU `dd89120` tests established bitwise legacy equivalence.
Compact tests independently expand attention into the full original grid and
compare against the masked reference, including action access restricted to frame 0.

The tests also check complete action/video scheduler counts, physically compact
Q/KV shapes, budgeted swaps, unchanged unselected cache rows, request/exception
isolation, poisoned replay buffers, changed observations, weight invalidation,
adapter overrides/fingerprints and frozen-profile re-import.

Evidence root:
`/root/wenbiao_zhao/dreamwam-sr/outputs/hybrid-implementation-20260920/`.
Logs: `c61cc89-cpu.log`, `5dec9c3-hybrid-cuda.log`.
Latest logs: `dc029e3-cpu.log`, `dc029e3-hybrid-cuda.log`.
The 46/512 manifests and two one-candidate control manifests were generated
successfully on the server.

Run the checks only on the server, using the existing verified Python:

```bash
HYBRID_PY=/root/wenbiao_zhao/dreamwam-sr/DreamWAM-fresh-6c52f36/.venv/bin/python
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONPATH=. "$HYBRID_PY" -m pytest tests/sparse -q
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONPATH=. "$HYBRID_PY" -m pytest \
  tests/sparse/test_hybrid_schedule.py tests/sparse/test_hybrid_runtime.py \
  tests/sparse/test_hybrid_compact.py tests/sparse/test_hybrid_graphs.py \
  tests/sparse/test_hybrid_policy.py tests/sparse/test_hybrid_experiment.py -q
"$HYBRID_PY" -m py_compile scripts/sparse/generate_hybrid_schedules.py \
  scripts/sparse/benchmark_hybrid_schedules.py scripts/sparse/summarize_hybrid_schedules.py
```

Physical GPU 3 admission was checked before each CUDA launch; GPU 5 was not used
by this work. These device numbers are evidence, not permission to skip fresh
admission checks on a later launch.

### Real-checkpoint finite/resume and numerical controls

Real-checkpoint debug verification uses source `5dec9c3`, physical GPU 3
(`GPU-c0af33a9-498c-ff7c-bb56-e9992ccded30`), checkpoint SHA256
`6c087e5b9e201f19dbe92834b470a7ee18937a18ea9adcbb49886aa1e3ac4c61`,
and the three already-exposed hash-verified real observations.
The bounded first invocation exited 0 at **3/24 timed requests** and correctly
reported PARTIAL with null paired speedup. Its accepted rows are preserved while
the second invocation accepted the remaining 21 rows, exiting 0 at **24/24**.
The independent all-Dense control completed **6/6**, exit 0. These runs ended at
12:32:15 UTC and 12:36:36 UTC, respectively. Commands/attempt timestamps are in
each `manifest.json`; stdout is in `compact-first.log`, `compact-resume.log`,
and `dense-control.log`. A compact provenance/hash snapshot is checked in as
[hybrid-debug-evidence-20260920.json](hybrid-debug-evidence-20260920.json).

Exact compact benchmark command, from the clean server worktree:

```bash
HYBRID_OUT=/root/wenbiao_zhao/dreamwam-sr/outputs/hybrid-implementation-20260920
HYBRID_INPUT=/root/wenbiao_zhao/dreamwam-sr/assets/outputs/head-stage-calibration-20260920/inputs-gpu4/manifest.json
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONPATH=. "$HYBRID_PY" scripts/sparse/benchmark_hybrid_schedules.py \
  --schedules "$HYBRID_OUT/compact-smoke.jsonl" --inputs "$HYBRID_INPUT" \
  --split-label debug --reps 2 --warmup 1 --group-size 1 \
  --max-new-requests 3 --trace --allow-shared-gpu --out-dir "$HYBRID_OUT/compact-run"
```

The second invocation replaced `--max-new-requests 3` with `--resume`.
The all-Dense command used `dense-control.jsonl`, `--reps 1`,
`--controls dense_strong`, no request limit/trace/resume, and output
`dense-control-run`. These are archived source-`5dec9c3` runs; the stricter
resume identity intentionally rejects trying to resume them from newer code.

The compact candidate uses D at step 0, S at step 5, R elsewhere, q=10%,
kv=25%, action-drift weight 1 and balanced frame quotas. This is an interface
control, not an empirically selected schedule. Each arm has six warm complete
requests (three historical observations × two repetitions), with common exact
prompt reuse and CUDA graph backend:

| Arm | Mean predict_action | Strong Dense / arm |
|---|---:|---:|
| Strengthened Dense | 262.24 ms | 1.000× |
| Legacy sparse refresh, full K/V reuse | 136.72 ms | 1.918× |
| Every-step fresh 10% | 216.69 ms | 1.210× |
| Hybrid sparse refresh, compact K/V reuse | 139.28 ms | 1.883× |

**No additional speedup over the old cached method:** legacy/hybrid is 0.982×
in this small debug run. Hybrid final-action relative L2 against Dense averages
0.831; this is a warning diagnostic, not a measured SR loss. The legacy arm uses
global queries whereas this compact arm uses balanced queries, so this comparison
does not isolate the K/V read factor. `afe6982` adds the matched-quota control
needed for that subsequent experiment.

The two-invocation run predates `attempt_id` instrumentation: its six pairs are
reported as **untracked** by the current read-only summary logic. The first
cyclic timing block crosses a restart. Raw journals are not rewritten to hide
that limitation. The shared-host admission flag, tiny sample count and historical
inputs further limit these numbers to implementation debugging, not a formal
performance or generalization result.

All 30 accepted action artifacts were independently reloaded: hashes and own-eager
bitwise parity passed, all requests performed 300 action-layer updates, and all
prompt-cache hits were recorded. The three all-Dense hybrid actions also matched
strengthened Dense bitwise (zero action error). That correctness control costs
306.39 ms versus 263.60 ms strengthened Dense in its separate run: the hybrid
wrapper is not an equally optimized Dense timing baseline. Keep the stronger
Dense denominator instead of inflating speedup with this slower control.

Three separate instrumented traces verified 294 dense Q/KV rows at step 0,
30 Q / 74 KV rows at step 5, and 0 visual Q / 74 KV rows at all eight reuse steps.
Every step executes all 30 action layers. Route replacement obeyed U ⊆ R and
new read keys ⊆ U; reuse routes were unchanged. The frozen profile re-imported
under `dc029e3` with identical plan hash and layout/scheduler compatibility.
Mean instrumented CUDA stream spans were 28.253 ms/dense step,
20.838 ms/sparse step and 8.326 ms/reuse step. They include staging/execution
stream gaps and are not a kernel-level attention/copy cost attribution.
The traces also retain selection, packing, preparation and commit CPU/CUDA spans;
do not sum CPU and CUDA spans or use the traced request as a latency sample.

### H100 OSMesa handoff and remaining work

The user's subsequent environment handoff reports H100 inference → CPU OSMesa →
LIBERO → persisted results working, with 256/256 renderer reads. Its existing
fresh-every-step configuration measured 262.81 → 216.73 ms and two task failures
at the 400-step limit; without matched Dense outcomes these do not establish an
SR loss. Those are a separate configuration/cohort and are not hybrid outcomes.
No 50-pair run was launched here, and none of its environment or controller
settings were modified by this implementation.

`deployment/h100/fresh-token-osmesa.env.sh` points `MODEL_ROOT` at the old frozen
fresh-token checkout. A future hybrid rollout must explicitly select a frozen
hybrid-capable checkout and verified Python, replace mutually exclusive
`fresh_visual_tokens` with `hybrid_visual`, preserve the matched prompt/protocol
settings, and use a new output manifest. Sourcing the fresh script alone does
not activate this code. Profile options are provided; a matched hybrid OSMesa
closed-loop cohort has not yet been configured and executed.

The complete 46/512 timing search, episode-disjoint selection, and 50-episode-per-arm
official SR screening remain separate experiments. The implementation and small
debug run do not establish a best refresh step, retained SR, or extra speed over
the existing cached method. Kernel-level cost attribution/copy-byte accounting
and closed-loop wall time also remain to be measured; current phase spans do
not substitute for that profiling. P1–P4's core code and P5's search/profile
interfaces are delivered, not P5's complete experimental acceptance.
