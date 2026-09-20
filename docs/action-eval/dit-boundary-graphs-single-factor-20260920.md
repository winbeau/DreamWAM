# DiT pre/post graph dispatch: no net full-request gain

Status: **MEASURED, negative for useful added acceleration**. Commit **5ac5c4d**
completed on **2026-09-20 00:45:55 UTC**, exit 0. Adding graph dispatch to the four
original DiT pre/post methods changes temporal full requests from **145.6188 to
145.6070 ms** (**1.000081×**) and native Dense graph requests from **286.9501 to
287.0528 ms** (**0.999642×**). The effect is negligible. The implementation stays
experimental and is not enabled in any SR policy.

## Fixed factor and execution

This follows the [small K/V staging factor](visual-kv-staging-single-factor-20260920.md),
but does not combine with it. The staging option is disabled, and the CLI rejects
combining these two new factors. Both controls retain the original visual K/V
copy schedule and existing transformer graph scope. Temporal keeps all visual
tokens at two refreshes (interval 5); native Dense performs all ten full calls
without exporting unused visual K/V.

[`DiTBoundaryGraphs`](../../dreamwam/sparse/dit_boundary_graphs.py) wraps the
original video/action `pre_dit` and `post_dit` methods separately. Every method
still executes at every denoising step. Fixed input buffers receive current
values; outputs are cloned. The video head receives the unchanged current time
embedding and grid size. A separate buffering-only control incurs the same
method-boundary buffering without graph replay, while retaining transformer
graphs. Context-manager composition restores both sets of methods on exit or
failure. Graph capture failures remain fatal, with no eager fallback.

Joint bf16, 30 layers, ten denoising steps, 294 visual tokens, nine video frames,
horizon 32, replan 10, seed 42, both original CPU RNG streams, both schedulers,
world residuals, preprocessing and weights are unchanged. Checkpoint SHA-256:
`6c087e5b9e201f19dbe92834b470a7ee18937a18ea9adcbb49886aa1e3ac4c61`.
The existing Python 3.10.20 / torch 2.7.1+cu126 environment was reused unchanged.
No dependency install or renderer workaround was introduced.

Frozen server worktree: `dreamwam-sr/DreamWAM-dit-boundary-5ac5c4d`. Tracked source
was clean; only the external `.venv` and `pretrained` symlinks were untracked and
recorded. The full sparse CPU suite passed **214 tests, five CUDA-only skips**,
in 5.95 s. All **five boundary tests passed with CUDA enabled** in 4.46 s,
including real Joint actions, world diagnostics, changed inputs, graph replay,
every-step execution, failed requests and restored methods. All test exits were 0.

## Full-request result

GPU 1 (`GPU-b9157129-ac58-501d-c93d-b1cf622ff61b`) was empty at launch: 0%
utilization, 1 MiB used, 143,156 MiB free. Earlier independently owned work had
exited; it was not signalled. This work did not allocate GPU 3. The benchmark
lasted **00:41:16–00:45:55 UTC**. Dense SR resumed separately on GPU 0 after its
preceding recovery batch reached its finite cap; temporal graph SR used GPU 2.

| Variant | Mean ms | Median ms | p95 ms |
|---|---:|---:|---:|
| Native Dense eager | 473.62 | 473.33 | 477.85 |
| Native Dense transformer graph | **286.95** | 284.99 | 303.38 |
| Dense graph + buffered pre/post | 287.96 | 286.15 | 304.44 |
| Dense graph + graphed pre/post | 287.05 | 285.08 | 303.47 |
| Temporal transformer graph | **145.62** | 144.40 | 155.42 |
| Temporal graph + buffered pre/post | 146.72 | 145.48 | 156.34 |
| Temporal graph + graphed pre/post | 145.61 | 144.38 | 155.57 |

Pre/post graph dispatch recovers about 0.90 ms for Dense and 1.11 ms for temporal
relative to the added buffering-only controls, but provides no useful gain over
the original transformer graph paths. The temporal/native ratio is **1.970557×**
before the factor and **1.971422×** after it. This separately supports the
[earlier native-Dense-control timing](temporal-graph-single-factor-20260919.md)
of **1.964983×** for conservative temporal reuse; it does not validate SR.

Each of 16 variants has two full-request warmups and 48 timing repetitions.
Each variant appears in each execution position three times, once per timing
input. All **768 timed requests** pass exact comparisons against their own eager
outputs, and parameter versions are unchanged. Initial correctness calls change
images, language and proprioception, then return to the first input. All warmed
timings include complete synchronized `predict_action` through CPU output;
synthetic inputs are not SR evidence. Every new timed variant executes all four
pre/post methods ten times, with 40 graph replays for the enabled variant and
zero boundary graph replays for the buffering control. The independent raw audit
verifies all 48 input/position pairs per variant, counts, means and these counters.
All timing modes and samples are retained without selecting a favorable window.

Model loading took 15.64 s. First observed full calls for buffered/graphed
boundaries were 0.552 / 0.761 s for Dense and 0.611 / 0.788 s for temporal.
They include setup and occur sequentially within one process, so they are
separate from warmed latency and are not equal-lifecycle startup comparisons.
Peak allocated/reserved memory across all benchmark variants was
26,877 / 27,826 MiB.

```bash
PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python scripts/sparse/benchmark_visual_cache_graphs.py \
  --include-partial-factor --include-temporal-control \
  --include-native-graph-control --include-dit-boundary-factor \
  --warmup 2 --graph-warmup 3 --reps 48 --out-dir "$NEW_OUTPUT_DIR"
```

Server artifacts: `dreamwam-sr/outputs/dit-boundary-factor-20260920/benchmark-5ac5c4d`.
The [local evidence bundle](evidence/dit-boundaries-20260920/) retains manifests,
source hashes, requests, summaries, test/exit logs, resource checks and raw audit.
No repeat or production integration is justified by this near-zero result.

## Full SR progress, separate from this experiment

At **2026-09-20 00:47:31 UTC**, independently audited official terminal records
and live controller handles were:

| Frozen full run | Terminal / planned | Successes | Task failures | GPU | Recovery supervisor |
|---|---:|---:|---:|---:|---:|
| Matched Dense | 344 / 500 | 337 | 7 | 0 | 213211 |
| Eager temporal | 142 / 500 | 142 | 0 | 1 | 217246 |
| Temporal graph | 108 / 500 | 108 | 0 | 2 | 215043 |

Dense batch 05 stopped at its 20-invocation cap with 328 outcomes (321 successes,
seven task failures), then batch 06 resumed. Temporal graph batch 01 stopped at
its cap with 98 official successes, then batch 02 resumed. GPU 1 became available
after this benchmark; eager temporal batch 04 resumed its unchanged run from 142
accepted successes. These use the frozen f8eae67 recovery helper, preserve every
accepted result hash, stop on three consecutive no-progress invocations or a
non-native error, and check GPU resources before each invocation. The evaluator
remains 4701ac2. No settled success or failure is retried, and no records from
different runs are pooled.

The guided 10% full run still retains 95 outcomes (93 successes, two failures)
and is inactive. The temporal graph's first 98 successes include both identities
where that guided run failed. This is a paired outcome observation, not proof of
the cause: first-camera-input repeatability is still unestablished. Full SR
remains null until complete coverage. Next: complete the frozen full matrices;
require measured cost evidence before another dispatch change.
