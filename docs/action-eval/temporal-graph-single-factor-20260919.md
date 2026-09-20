# Temporal graph reuse and a stronger Dense control

Status: **MEASURED; full paired SR pending**. The final control completed on
**2026-09-20 at 00:07:51 UTC**, exit 0: conservative temporal reuse takes
**149.60 ms** per complete request versus **293.96 ms** for native Dense with
the same transformer graph scope, **1.964983×**. The 10% token candidate takes
**143.70 ms**, **2.045713×** against this stronger Dense control. Neither number
establishes preserved benchmark quality. The conservative graph pilot completed
15/15 official successes; its independent 500-identity Spatial run is underway.

Continuation: a [separate K/V copy-schedule factor](visual-kv-staging-single-factor-20260920.md)
passed 1,008 timed parity checks but added only 0.86% temporal speedup in its
balanced confirmation under shared load. It remains experimental; the active
full SR run below still uses its original graph copy schedule.

## Separate factors and fixed conditions

This follows the [dense/action and partial-refresh graph factors](visual-graph-dispatch-single-factor-20260919.md).
Commit **a6910f3** adds a full visual-refresh budget control at fixed refresh
interval 5 and fixed all-transformer graph scope. Its eager reference is the
original `VisualStepCache`, the implementation used in the frozen V1 SR run.
Keeping all 294 visual tokens skips selection and action-mass construction;
two dense visual refreshes and eight action-only reuse steps remain. This is a
conservative temporal ablation, not evidence for action-guided token selection.

Commit **c8da19a** adds a separate
[`NativeDenseGraph`](../../dreamwam/sparse/dense_graph_control.py) benchmark
control. It replays the original full Joint transformer and returns its outputs
and current world-router diagnostics. It does not capture/export visual K/V
that a full-Dense request never reuses. The earlier full-budget cache wrapper
remains in the comparison, so its overhead is measured explicitly. This control
has no production policy option and does not alter any live SR worker.

Both factors retain the checkpoint SHA-256
`6c087e5b9e201f19dbe92834b470a7ee18937a18ea9adcbb49886aa1e3ac4c61`,
Joint bf16, ten denoising steps, 30 layers, 294 visual tokens, nine video frames,
horizon 32, replan 10, seed 42, original CPU RNG, schedulers, preprocessing and
world residuals. The existing Python 3.10.20 / torch 2.7.1+cu126 environment was
reused unchanged; no install, sync or renderer workaround was performed.

## Complete request measurements

Both runs used GPU 2 (`GPU-b15ccd2e-b17d-20a9-a130-b4b7eb37a074`), empty at
launch. GPU 3 was not allocated by this work. Each variant receives two warmed
full calls, followed by 30 repetitions in rotating variant order. Timing includes
synchronization and all of `predict_action` through CPU action output. The three
synthetic random-image inputs each occur ten times. An additional correctness
request changes language and proprioception, then returns to the original input.
These inputs provide latency/parity evidence, not SR evidence.

| Variant | a6910f3 mean ms | c8da19a mean ms | c8da19a median ms | c8da19a p95 ms |
|---|---:|---:|---:|---:|
| Native Dense eager | 463.97 | 471.23 | 470.52 | 476.31 |
| Matched Dense eager | 469.15 | 473.71 | 473.47 | 475.65 |
| Guided 10% eager | 281.91 | 284.70 | 284.60 | 286.19 |
| Temporal full-refresh eager | 282.98 | 285.78 | 285.98 | 287.30 |
| Full-budget Dense graph wrapper | 298.41 | 296.00 | 288.48 | 307.70 |
| Guided 10%, all transformer graphs | 145.07 | 143.70 | 139.20 | 150.95 |
| Temporal full-refresh graphs | 151.04 | 149.60 | 145.09 | 156.80 |
| Native Dense graph, no unused K/V export | — | **293.96** | 286.48 | 306.18 |

The full-budget wrapper overhead is **1.006929×** relative to native Dense graph
(about 2.04 ms). Accounting for it changes temporal speedup from **1.978598×** to
**1.964983×**, and guided speedup from **2.059888×** to **2.045713×**. The token
budget factor alone gives **1.041085×** (5.90 ms) beyond temporal graph reuse.
The conservative path's graph factor versus its own eager implementation is
**1.910273×**. Its combined graph-plus-cache ratio against unmodified Dense is
**3.149952×**; that combined ratio must not be attributed solely to sparsity.

The first run lasted 23:50:19–23:52:56 UTC on September 19 and recorded 330 timed
requests. The stronger control lasted 00:05:02–00:07:51 UTC on September 20 and
recorded 360. All 690 timings passed bitwise comparisons against their own eager
path, including graph temporal against original V1 and graph Dense against
native Dense. Every temporal timed call executed two dense graph replays,
eight action graph replays and all 300 action layer updates, with no partial
graph replay. Native Dense exported zero visual-cache layers. Parameter versions
were unchanged. An independent raw-record audit recomputed all sample counts,
input balance and mean latencies. All samples, including the two timing modes in
graph variants, were retained; no best-window subset was selected.

In the final run model loading took 17.68 s. First observed full calls were
0.738 s for temporal graph, 1.066 s for guided all-graph, and 0.687 s for native
Dense graph. They occurred sequentially inside one process and are not equal
startup-lifecycle comparisons. Internal graph setup took 0.368 + 0.225 s for
temporal dense/action graphs and 0.354 s for native Dense. Peak allocated and
reserved memory across all variants was 25,766 / 26,348 MiB. The c8da19a sparse
test suite passed **206 tests, 2 CUDA-only skips, in 5.67 s**, exit 0; its real
checkpoint benchmark then exercised CUDA capture directly. Earlier policy/API
verification at 2c02c5c separately passed all 206 tests with CUDA enabled.

```bash
# Frozen worktree DreamWAM-native-graph-c8da19a; unchanged existing environment.
PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=2 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python scripts/sparse/benchmark_visual_cache_graphs.py \
  --include-partial-factor --include-temporal-control \
  --include-native-graph-control --warmup 2 --graph-warmup 3 --reps 30 \
  --out-dir "$NEW_OUTPUT_DIR"
```

For a6910f3, omit `--include-native-graph-control`. Authoritative server artifacts
are under `dreamwam-sr/outputs/temporal-graphs-20260919/benchmark-a6910f3` and
`dreamwam-sr/outputs/native-graph-control-20260919/benchmark-c8da19a`.
The [local evidence bundle](evidence/temporal-graphs-20260919/) retains both raw
timing sets, manifests/source hashes, summaries, exit codes, tests, resource
inventories and the independent raw audit.

## Official pilot and full coverage

The pilot uses model worktree **2c02c5c**, configuration **3b1bf18**, and frozen
evaluator **4701ac2**. Its executed fingerprint confirms keep ratio 1.0,
refresh every 5, guidance weight 1.0 and `graph_dispatch: all_transformers`, with
the unchanged checkpoint and official protocol. The literal guidance setting
does not build action mass at full visual budget, as the benchmark counters show.
Config fingerprint:
`bde052cbb32f5620655bcd2ff1e5a16959c66c0df89200124fa0ffe1f51de756`.
`validate` and `doctor` both exited 0 before launch; doctor is not a model smoke.

Run `temporal-graph-pilot-20260919/gpu2/run-20260919T235542Z-bde052cb` completed
all 15 planned task/init identities, each an official success. Three bounded
recoveries completed coverage after native EGL aborts; the recovery exit was 0.
Pairing against Dense pilot `run-20260919T210051Z-93e56600` gives 15 shared
successes and no discordances. This remains a 15-episode pilot. All 15 first
policy state vectors matched Dense, but neither camera hash matched on any of
the 15 pairs. Hash differences do not measure image-error magnitude or establish
a cause; repeatable initial observations have not been demonstrated.

The independent 500-identity run is
`temporal-graph-libero-spatial-20260919/gpu2/run-20260920T001038Z-cf9e4f10`, with
config fingerprint
`cf9e4f105affc2b519291185bba90be7b8675dd139846b78bda133729afba1a9`.
Fresh launch validation/doctor/resource checks passed. GPU 2 shares the host
under the explicit sharing configuration; other users' processes remain intact.
Its initial six official successes were followed by `read_pixels` native abort
(exit 134). Recovery uses the frozen f8eae67 helper, at most 20 invocations per
batch and three consecutive no-progress invocations, preserving settled results.

At **2026-09-20 00:12:16 UTC**, independently audited terminal coverage was:

| Separate run | Terminal / planned | Official successes | Official task failures |
|---|---:|---:|---:|
| Matched Dense full | 274 / 500 | 272 | 2 |
| Eager temporal full | 142 / 500 | 142 | 0 |
| Eager guided 10% full | 95 / 500 | 93 | 2 |
| Graph temporal full | 6 / 500 | 6 | 0 |

Dense's failures `t005-i002-r00-s42` and `t005-i003-r00-s42` remain accepted task
outcomes. Guided's two failures are on different identities, `t000-i048-r00-s42`
and `t001-i036-r00-s42`, where Dense and eager temporal succeeded. All record
hashes are retained. Dense batch 05 and graph temporal batch 01 are the active
continuations. Eager temporal stopped at a GPU resource gate after 142 outcomes;
GPU 1 is occupied by an independently owned evaluation. Guided remains inactive.
No SR is reported for incomplete full runs; no counts from separate runs or
pilots are pooled. Next: complete the frozen Dense/temporal graph matrix, compare
official paired outcomes, and keep further dispatch optimizations as separate
factors with the native Dense graph control.
