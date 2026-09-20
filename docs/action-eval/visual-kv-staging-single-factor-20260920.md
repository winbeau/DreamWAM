# Repeated visual K/V copies: small confirmed dispatch gain

Status: **MEASURED, experimental only**. Commit **d77735c** removes repeated
copies of read-only visual K/V between visual refreshes. In the balanced
confirmation completed **2026-09-20 00:30:54 UTC**, exit 0, conservative temporal
requests changed from **161.05 to 159.68 ms**, **1.008587×** (1.37 ms). The first
run's larger **1.030624×** gain did not persist at the same magnitude. Guided 10%
improved by **1.008492×** initially and **1.008239×** in confirmation. The factor
remains benchmark-only; no active SR policy or experiment configuration changed.

## One factor, fixed sampling

This builds on [temporal graph reuse and the native Dense graph control](temporal-graph-single-factor-20260919.md).
[`StagedVisualKVCache`](../../dreamwam/sparse/visual_kv_staging.py) changes only
the input-copy schedule of the action transformer graph. The original
`forward_action_with_video_cache` reads visual K/V without modifying it. An
explicit generation advances at every request boundary and dense/partial visual
refresh; the first action call after a refresh copies all current per-layer K/V.
Later action calls in that phase reuse those staged tensors. No tensor-address or
mutation-version heuristic decides freshness. Action tokens, context, timestep
modulation and all other inputs are copied on every call. Input structure,
shape, dtype, device, stride and metadata are still validated, and graph outputs
still do not alias buffers exposed to the caller.

The implementation leaves graph scope, exact kernels, selection, all schedulers,
both CPU RNG streams, precision and weights unchanged. Temporal uses keep ratio
1.0, interval 5; guided uses 0.1, interval 5, weight 1.0. Both use all-transformer
capture. All ten action steps remain. The checkpoint SHA-256 remains
`6c087e5b9e201f19dbe92834b470a7ee18937a18ea9adcbb49886aa1e3ac4c61`.
Native Dense graph remains the strongest unchanged control, with no unused visual
K/V export. The factor's unmodified temporal/guided graph controls are retained.

## Tests and full requests

Frozen server worktree: `dreamwam-sr/DreamWAM-kv-staging-d77735c`. The existing
Python 3.10.20 / torch 2.7.1+cu126 environment was reused unchanged. Tracked source
was clean; the only untracked entries were the `.venv` and `pretrained` external
symlinks, recorded explicitly rather than hidden. An initial deployment guard
stopped on those two entries before running tests; no source modification was
made on the server.

The full sparse CPU suite passed **211 tests with three CUDA-only skips** in
6.83 s. All **six staging tests passed with CUDA enabled** in 4.47 s. These cover
changed dynamic inputs, in-place partial refresh at the same tensor address,
request boundaries, real Joint sampling, eager versus graphed partial refresh,
restored methods and actual CUDA replay. All test exits were 0.

GPU 4 (`GPU-490b4a76-6210-31b9-4e03-838a113cf5f4`) was explicitly shared. The
initial timing launch observed 8% utilization, 96,907 MiB used and 46,250 MiB free;
confirmation observed 11%, 96,947 MiB used and 46,210 MiB free. Other users'
processes were not signalled, and this work did not allocate GPU 3. Graph timings
have substantial variation under this load. Absolute values must not be pooled
with earlier empty-GPU-2 results.

| Variant | Initial mean ms | Confirmation mean ms | Confirmation median ms | Confirmation p95 ms |
|---|---:|---:|---:|---:|
| Native Dense eager | 465.21 | 458.10 | 457.97 | 462.57 |
| Native Dense graph | 321.90 | 321.85 | 327.79 | 332.83 |
| Temporal graph, original copies | 164.46 | 161.05 | 165.93 | 168.25 |
| Temporal graph, refresh-scoped copies | 159.58 | 159.68 | 163.19 | 166.69 |
| Guided graph, original copies | 153.25 | 153.22 | 155.83 | 161.71 |
| Guided graph, refresh-scoped copies | 151.96 | 151.97 | 154.62 | 160.30 |

The staged temporal/native graph ratios are **2.017229×** and **2.015655×**;
guided ratios are **2.118359×** and **2.117870×**. Those observed ratios do not
establish a stable 2× guarantee: even the unmodified temporal/native graph ratio
varies with the shared load. The incremental staging gain in confirmation is
only 0.86%, while its per-request timing standard deviation is 8.21 ms.

All **420 initial and 588 confirmation timed requests** passed exact comparisons
against their own eager outputs, and parameter versions were unchanged. Four
correctness inputs change images, instruction and proprioception, then return to
the first input; three synthetic image inputs are used for timing, never for SR.
Two full-request warmups precede each variant's timing. All samples are retained.
Each timed staged request copied K/V twice and skipped six batches, retaining
eight action graph calls and 300 action layer updates. Counters independently
confirmed **216,760,320 copied bytes and 650,280,960 skipped bytes per request**.

The confirmation uses 42 repetitions for 14 variants: every variant appears in
every position exactly three times, once for each of the three timing inputs.
Independent raw audits verify all 42 input/position pairs for every variant,
counts, parity flags, staging counters and recomputed means. The change in
repetition count responds to the observed load variation; code and factors are
unchanged. No further repeat was used to select a favorable result.

Initial model loading took 14.62 s. First observed full staged temporal calls
were 3.291 s initially and 0.626 s in confirmation; staged guided calls were
1.048 and 0.932 s. Native Dense graph first calls were 1.293 and 0.568 s. These
sequential first calls include capture and shared-host effects and are separate
from warmed means, not equal-lifecycle startup comparisons. Initial peak
allocated/reserved memory across all variants was 26,569 / 27,332 MiB.

```bash
PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=4 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python scripts/sparse/benchmark_visual_cache_graphs.py \
  --include-partial-factor --include-temporal-control \
  --include-native-graph-control --include-kv-staging-factor \
  --warmup 2 --graph-warmup 3 --reps 42 --out-dir "$NEW_OUTPUT_DIR"
```

Initial timing used `--reps 30`. Both exited 0. Server artifacts are under
`dreamwam-sr/outputs/kv-staging-factor-20260920/{benchmark,confirmation}-d77735c`;
the [local evidence bundle](evidence/kv-staging-20260920/) retains manifests,
source hashes, requests, summaries, test/exit logs, launch inventories and audits.

## SR remains a separate acceptance condition

This exact-copy factor has no standalone SR run and is not claimed to repair the
guided candidate's task failures. The active conservative full run still uses
model **2c02c5c**, config **3b1bf18**, evaluator **4701ac2**, and the original copy
schedule. Its first 50 task-0 identities all succeeded. At 00:25 UTC, Dense had
298/500 terminal outcomes (293 successes, five task failures), while temporal
graph had 50/500 successes. Both controllers were live. Their separate
[progress and executed-policy audits](evidence/temporal-graphs-20260919/full-progress/)
retain record hashes and verify ten completed worker fingerprints. A live
provenance file briefly lacks the description during worker startup; the
completed handover snapshots confirm the actual options. This is missing
in-flight metadata, not an option mismatch. No incomplete full-run SR is reported.

Next: prioritize completing the frozen Dense/temporal graph coverage. Keep the
small staging gain experimental; any next dispatch factor must be measured
separately with the same native Dense optimization.
