# Transformer graph dispatch: isolated full-request measurements

Status: **MEASURED**. Adding partial-refresh capture at **e9c50e3** completed
**2026-09-19 23:28:14 UTC**, exit 0: **142.12 ms** guided versus **293.29 ms**
equally graphed matched Dense, **2.063626×**. It follows the separately measured
dense/action graph factor at **62ce404**, completed **23:21:30 UTC**, exit 0,
which measured **1.856882×**. Every timed buffered/graphed output is bitwise its
own eager output. This is latency and implementation evidence; complete paired
SR is still pending. Neither graph factor changes the ongoing full SR runs.

## Fixed conditions and implementation

This follows the [action-guided visual-token factor and diagnostic profile](visual-token-cache-single-factor-20260919.md).
The checkpoint remains
`6c087e5b9e201f19dbe92834b470a7ee18937a18ea9adcbb49886aa1e3ac4c61`.
Joint, bf16, 294 visual tokens, 32 actions, 30 layers, ten denoising steps, nine
video frames, horizon 32, replan 10, seed 42 and the original separate CPU random
generators are unchanged. Guided selection remains keep ratio 0.1, refresh every
five steps, action guidance weight 1.0. No compile, fused fast-ops, weight change,
precision change, renderer workaround or dependency install is introduced.

[`GraphedVisualTokenCache`](../../dreamwam/sparse/visual_cache_graphs.py) extends
the existing implementation. At the first factor it captures only full Joint
transformer calls and action-only transformer calls with cached visual K/V.
The dense control captures all ten full calls; guided captures the dense anchor
and eight reuse calls, while its one partial refresh stays eager. Selection,
all pre/post-DiT work, both schedulers, text/VAE processing and CPU action output
remain in the full request. Cache statistics count logical model work, excluding
the extra transformer warmups used only for graph setup.

Every input is filled into fixed buffers for each invocation, including current
context, timestep modulation and all per-layer visual K/V. Each output is cloned
before it can be mutated by a later partial refresh. Graph replay does not execute
Python hooks, so current visual K/V and world-router diagnostics are explicitly
restored from graph products. Request-local cache selection is still cleared at
every sampling boundary. Capture failure raises an error instead of silently
timing an eager fallback. These choices follow the fixed-address and CPU-side
execution constraints in the [PyTorch 2.7 CUDA Graph documentation](https://docs.pytorch.org/docs/2.7/notes/cuda.html#cuda-graphs).

Commit **9e540cc** additionally exposes strict model-side policy options; its
deployment verification is described below. Existing full SR workers remain
pinned to their original worktrees.

## First factor: dense and action transformer capture

Server checkout: `dreamwam-sr/DreamWAM-graphs-62ce404`, Git clean, existing
Python 3.10.20 / torch 2.7.1+cu126 environment reused unchanged. GPU 2 UUID
`GPU-b15ccd2e-b17d-20a9-a130-b4b7eb37a074` was empty at launch, 0% utilization,
1 MiB used. GPU 3 was not allocated by this work. Other cards had independently
owned jobs and were not signalled.

The complete sparse test directory passed **193 tests in 7.38 s**, exit 0.
The real-checkpoint benchmark then checked three synthetic image inputs and an
additional request changing language/proprio, followed by a return to the first
input. All seven paths passed bitwise checks against their own eager policy,
including the full-budget path against native Dense. Every one of the subsequent
210 timed requests also passed, and parameter versions were unchanged.

Two full-request warmups per variant precede 30 rotating-order repetitions per
variant. Each sample includes a complete synchronized `predict_action` through
CPU action output. The latency inputs are the same three synthetic random image
seeds 0/1/2 and zero proprio used for earlier factors; they are not SR evidence.

| Variant | Mean ms | Median ms | p95 ms |
|---|---:|---:|---:|
| Native Dense | 463.82 | 463.50 | 467.18 |
| Eager matched Dense | 470.14 | 469.74 | 473.66 |
| Eager guided | 282.53 | 282.18 | 284.46 |
| Static-buffer Dense, no graph | 479.41 | 479.39 | 481.99 |
| Static-buffer guided, no graph | 287.96 | 287.54 | 289.77 |
| Graphed matched Dense | **291.93** | 288.82 | 307.79 |
| Graphed guided | **157.22** | 155.55 | 165.44 |

Graph dispatch accelerates guided by **1.797093×** relative to its eager path and
Dense by **1.610446×**. The visual-cache ratio with equally optimized controls is
**1.856882×**, versus 1.664024× for the two eager paths in this same run. Against
unmodified native Dense, the combined cache-plus-graph result is **2.950167×**;
this combined figure must not be presented as the visual approximation's speedup
over equally optimized Dense. Buffering alone is slower for both paths.

The first five graphed samples were around 307–308 / 165–166 ms; later samples
were around 288–289 / 155–156 ms. All samples remain in the result. Rotating the
seven variants helps balance load drift but does not make absolute host/GPU
timing invariant. No best-window subset was selected.

Model loading took 16.68 s. First observed full calls, retained separately from
warm timing, were 1.207 s for graphed Dense and 0.549 s for graphed guided; the
first native call was 0.930 s. These first calls occur sequentially within one
loaded process, so they are not equal-startup-lifecycle comparisons. The internal
graph setup spans were 0.898 s for Dense and 0.250 + 0.153 s for guided's dense
and action graphs. Input-buffer creation is included in the first full call,
but not in those narrower setup spans. Peak allocated/reserved CUDA memory across
all benchmark variants was 24,698 / 24,888 MiB.

```bash
PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=2 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python scripts/sparse/benchmark_visual_cache_graphs.py \
  --warmup 2 --graph-warmup 3 --reps 30 --out-dir "$NEW_OUTPUT_DIR"
```

Server artifacts: `dreamwam-sr/outputs/visual-graphs-20260919/benchmark-62ce404`.
Committed evidence: [manifest and exact command](evidence/visual-graphs-20260919/dense-action/manifest.json),
[summary](evidence/visual-graphs-20260919/dense-action/summary.json),
[all 210 requests](evidence/visual-graphs-20260919/dense-action/requests.jsonl),
[tests](evidence/visual-graphs-20260919/dense-action/tests-62ce404.log),
[launch GPU inventory](evidence/visual-graphs-20260919/dense-action/launch-gpus-62ce404.csv).

## Second factor: capture partial-refresh tensor math

Commit **e9c50e3** adds opt-in `graph_partial=True` and the benchmark flag
`--include-partial-factor`. The already measured dense/action capture remains an
explicit control. Only partial-refresh tensor math enters the added graph; its
selected indices and all anchor K/V are current-call inputs, and updated K/V are
copied back. Every capture warmup and first replay refills mutable K/V inputs
from the original anchor. A separate full-budget control enables the same flag.
Action guidance and selection still run eagerly with the same budget/cadence.

The second server run used frozen checkout `DreamWAM-graphs-e9c50e3` on the same
GPU 2, again empty at launch. **196 sparse tests passed in 5.63 s**, exit 0.
The same four correctness inputs and return-to-first-input check passed for all
nine variants. All **270 timed full requests** were bitwise their own eager
output. Weights and parameter versions remained unchanged.

| Variant, same second run | Mean ms | Median ms | p95 ms |
|---|---:|---:|---:|
| Native Dense | 454.63 | 454.42 | 456.94 |
| Eager matched Dense | 460.26 | 460.16 | 463.26 |
| Eager guided | 277.12 | 276.78 | 279.45 |
| Static-buffer Dense, no graph | 467.76 | 467.18 | 471.15 |
| Static-buffer guided, no graph | 282.77 | 282.58 | 286.52 |
| Dense/action graphed Dense | 293.20 | 288.35 | 306.88 |
| Dense/action graphed guided | 156.61 | 154.19 | 163.31 |
| All-transformer graphed Dense | **293.29** | 288.44 | 306.58 |
| All-transformer graphed guided | **142.12** | 138.84 | 150.05 |

Partial-refresh capture contributes **1.101934×** on guided, while the full-budget
control ratio is 0.999701×, effectively unchanged. The guided-to-matched ratio is
**2.063626×**. The combined result versus unmodified native Dense is **3.198886×**,
which includes the common dispatch optimization and must be labelled accordingly.
These are full-request means, not sampler-only timings or selected best samples.

The all-transformer guided cache executed one dense replay, eight action replays
and one partial replay per request. The original 9,720 computed visual token-layer
rows out of 88,200, selected indices and ten logical action steps are unchanged.
The first complete guided graph call was **2.316 s**, including setup; graph setup
spans were 0.335 / 0.797 / 1.063 s for dense/action/partial capture. These costs are
not hidden in the warm result. Peak allocated/reserved memory over all nine
variants was 25,365 / 25,764 MiB. The same early/later graph timing variation is
visible in the retained raw samples.

Command: the first command above with **`--include-partial-factor`** and a new
output directory. Server artifacts:
`dreamwam-sr/outputs/visual-graphs-20260919/benchmark-e9c50e3`.
Evidence: [manifest and exact command](evidence/visual-graphs-20260919/partial-refresh/manifest.json),
[summary](evidence/visual-graphs-20260919/partial-refresh/summary.json),
[all 270 requests](evidence/visual-graphs-20260919/partial-refresh/requests.jsonl),
[tests](evidence/visual-graphs-20260919/partial-refresh/tests-e9c50e3.log).

## Policy integration and remaining acceptance

The model accepts `visual_cache.graph_dispatch: dense_action` or
`all_transformers`, alongside the existing ratio/cadence/guidance settings.
Omission preserves the existing eager configuration and fingerprint exactly.
Unknown values and CPU graph requests fail before loading weights. The adapter
continues to pass model options through and report the model's effective config
and actual per-request replay counters. `close()` releases graph buffers.

At **9e540cc**, CPU tests passed **204, with 2 CUDA tests skipped**. The first
targeted CUDA test run failed 2 of 3 cases before graph execution: the test fixture
constructed its tiny model on CPU then moved parameters, leaving unregistered
complex RoPE tables on CPU. Production `runtime.build_model` constructs inside
the selected device context. The fixture was corrected at **2c02c5c**. The
complete sparse suite then passed **206 tests in 10.37 s**, exit 0, with GPU 2
visible. Both graph modes executed through the real tiny Joint sampler and thin
adapter, reproduced eager actions, reported actual replay counts and released
their graph buffers on close. Failed test evidence is retained alongside the
[passing CUDA-enabled suite](evidence/visual-graphs-20260919/policy/tests-2c02c5c-cuda.log).

A separate 15-identity pilot began **23:35:14 UTC** on GPU 2 with model
**2c02c5c**, config **05d97df**, and frozen evaluator **4701ac2**:
`dreamwam-sr/outputs/visual-graph-pilot-20260919/guided-gpu2/run-20260919T233514Z-a1a810b6`.
[Validation](evidence/visual-graphs-20260919/pilot-preflight/validate.log) and
[doctor](evidence/visual-graphs-20260919/pilot-preflight/doctor.log) returned 0.
At **23:36:39 UTC**, it had four accepted successes and no task failures; the
actual worker fingerprint declared `graph_dispatch: all_transformers`, the
unchanged checkpoint and all scientific settings. It later exited 134 in the
existing native renderer path; a bounded recovery batch was started. This is
an independently recorded plumbing pilot, not a full SR result.

By **23:43:30 UTC**, recovery 3 had completed all **15/15 pilot episodes**,
all successful, exit 0. The [official paired pilot](evidence/visual-graphs-20260919/pilot-complete/paired-pilot.json)
contains 15 paired successes against the prior Dense pilot. The
[first-input audit](evidence/visual-graphs-20260919/pilot-complete/initial-input-audit.json)
again finds 15 equal state hashes and zero equal image hashes for either camera.
This small pilot does not establish non-inferiority and does not include the two
later-index identities that failed in the full eager guided run. No pilot records
were imported into a full run. [Per-episode records](evidence/visual-graphs-20260919/pilot-complete/per_episode.csv),
[provenance](evidence/visual-graphs-20260919/pilot-complete/provenance.json) and
[recovery completion](evidence/visual-graphs-20260919/pilot-complete/recovery-summary.json)
are retained.

The same [official-record snapshot](evidence/visual-graphs-20260919/pilot-preflight/progress-2336Z.json)
also found **95/500** terminal outcomes for eager guided (93 successes and
2 task failures), **208/500** for matched Dense (208 successes), and **118/500**
preserved temporal-only outcomes (118 successes). All incomplete rates remain
null. The two guided failures were `t000-i048-r00-s42` and `t001-i036-r00-s42`,
both at the official step limit with 40 policy calls. Dense and temporal-only
succeeded on both identities. Their accepted result hashes and termination
reasons are in the snapshot; these task failures are not retried or discarded.

Guided's first recovery supervisor ended at its 20-attempt cap. After verifying
its worker/supervisor had exited and GPU 1 was empty, allocation returned to the
more conservative temporal-only full run from its preserved 118 outcomes.
Guided's 95 outcomes remain inactive and unchanged. This prioritizes SR evidence;
the initial-image variability still prevents attributing the two discordances
solely to visual selection. Graph replay is bitwise the same guided method on
the tested inputs and is not a remedy for those task failures.

At **23:43:30 UTC**, [coverage and controller liveness](evidence/visual-graphs-20260919/pilot-preflight/progress-2343Z.json)
were re-audited: Dense **221/500** successes and temporal-only **132/500**
successes, with their supervisors alive; guided remains **93 successes + 2 task
failures / 500**, inactive. Worker PIDs may disappear between native recoveries;
the recorded supervisor handles and UTC timestamp define this snapshot.

An [independent artifact audit](evidence/visual-graphs-20260919/raw-evidence-audit.json)
recomputed every variant's mean from all 210/270 samples, checked ten samples
per synthetic input per variant, bitwise flags and executed replay/token counts,
and recorded the original artifact SHA-256 values. It passed without dropping
samples or changing either report.

Next: complete matched Dense and the temporal-only full run; measure graph
dispatch on the temporal-only control to
characterize its speed/quality tradeoff. Continue the 10% full run when capacity
permits, with all settled successes and failures retained. The first-image
repeatability limitation and native EGL errors remain unresolved, and full SR
acceptance remains open.
