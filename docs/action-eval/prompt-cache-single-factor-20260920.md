# Exact prompt reuse on matched Dense and guided visual sparsity

Status: **VERIFIED latency and parity; complete SR pending**. Date: **2026-09-20 UTC**.
The user prioritizes a working sparse method above further Head/Stage kernel
experiments. The main sparse candidate remains action-guided whole-transformer
token recomputation: first-step dense anchor, 30/294 visual tokens refreshed at
step 5, eight reused steps, and all 300 action layer updates. This executes
9,720/88,200 visual token-layer updates (11.02%, including the anchor), rather
than restricting a dense attention mask and calling it faster execution.

The one new common factor memoizes exact prompt batches in the frozen text
encoder. Dense receives the same optimization. Images, state, sampler RNG,
weights, bf16 precision, horizon 32, replan 10 and ten denoising steps remain
unchanged. The Dense control already reuses the invariant conditioned frame
and uses transformer graphs; Sparse retains r5 / 10% / action guidance 1 and
the same existing transformer-graph implementation.

## Measured complete requests

Run **2026-09-20 05:02:29–05:03:56 UTC**, exit **0**, source
`27c081e54cbb92aeb905ae263bc6edf44d86c6ed`, GPU **4**
(`GPU-490b4a76-6210-31b9-4e03-838a113cf5f4`). The target card was empty at
admission; a low-utilization compute-only card was left unused by this task.
Other jobs continued on the shared host. This is a warm full-`predict_action`
measurement through CPU action output, including image preprocessing and VAE.

| Variant | Mean ms | p50 ms | p95 ms | Speedup vs matched Dense |
|---|---:|---:|---:|---:|
| Strengthened Dense, uncached instruction | 269.838 | 269.815 | 270.141 | — |
| Guided token sparsity, uncached instruction | 150.159 | 150.142 | 150.622 | 1.797× |
| Strengthened Dense, cached instruction | 257.205 | 257.220 | 257.429 | — |
| Guided token sparsity, cached instruction | **137.370** | **137.386** | **137.520** | **1.872×** |

The new common factor improves Dense by **1.049×** and Sparse by **1.093×**.
Its benefit is separate from sparse execution. Sparse still executes 9,720 visual
token-layer updates per request, versus 61,740 for the strengthened Dense and
88,200 for original full recomputation. All arms execute **300 action layer
updates**. No Head × Stage compact attention is involved.

The **96** timed requests have 24 observations per variant, balanced over three
real inputs and four order positions: all 48 variant/input/position cells contain
two observations. All 96 timed outputs and 20 additional graph/cache comparisons
equal their own uncached eager actions bitwise. Real text contexts and masks also
passed exact comparisons and returned-tensor mutation checks. These checks do
not imply Sparse equals Dense or establish official SR preservation.

Six explicit first-instruction misses with graphs already warm average **271.034
ms Dense / 151.353 ms Sparse (1.791×)**. Arithmetic amortization of one such miss
plus nine measured hits gives **1.863×** over ten requests; this is an estimate,
not measured episode wall time. Initial graph-setup requests are retained too:
Dense raw **1,164.362 ms**, Sparse raw **773.741 ms**. Cached variants in that
sequence reused those same graphs, so their first logged requests are not cold
graph measurements. The 1.872× headline excludes initial graph capture/model load.

## Implementation and protocol

Memoization has a bounded eight-entry LRU, exact strings/order/batch keys,
fresh tensor copies on return, and invalidation on encoder parameter/buffer,
device, dtype or tokenizer configuration changes. It requires frozen eval
weights. It never caches observations or actions. Cold instruction misses
are charged separately, and no miss is relabelled a warm hit.

`benchmark_prompt_cache.py` used the three hash-verified real calibration
observations and an additional parity-only input that combines another real
image/state with a repeated instruction. It compares four variants: Dense
and Sparse, each with and without memoization. Twenty-four rotating repetitions
per variant balance three real inputs and four positions (96 timed requests).
All graph/cache actions must equal their own uncached eager reference bitwise.
Direct real-encoder checks compare context and mask tensors and deliberately
modify returned copies to verify ownership. Six forced misses with warm graphs
report first-instruction cost. The full boundary includes preprocessing, VAE,
all sampling steps and CPU action output; complete paired SR remains pending.

Checkpoint SHA256:
`6c087e5b9e201f19dbe92834b470a7ee18937a18ea9adcbb49886aa1e3ac4c61`.
The existing Python 3.10.20 / torch 2.7.1+cu126 server environment was reused
without installs or sync. Server CPU tests: **7 passed in 1.57 s**, exit 0;
benchmark script compilation also exited 0. These are the pre-integration tests
at the measured revision, not a claim about later adapter changes.

Exact benchmark command in the frozen source worktree:

```bash
PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python scripts/sparse/benchmark_prompt_cache.py \
  --inputs "$OUT/head-stage-calibration-20260920/inputs-gpu4/manifest.json" \
  --out-dir "$OUT/prompt-cache-20260920/factor-27c081e"
```

`$OUT` denotes the recorded server outputs root. Raw manifest, 96-request journal,
summary, admission, log, exit and the **05:08:26 UTC independent audit** are
[preserved here](evidence/prompt-cache-20260920/). The audit recomputed all means,
checked balanced coverage, outputs, executed budgets, cache hits and checkpoint
identity, and confirmed both collector/wrapper PIDs had exited. Retrieved file
hashes match the server audit. Manifest SHA256:
`762b618cb80628b557005b9d2142a24dc976bd36724817bbcf6d8027b20345cb`;
journal SHA256: `631ec91ff4d9cec7261720aa8009adbd97700233bb0517c4b1017488afa4d1a6`.
The preceding compact Head/Stage negative result remains
[fully preserved](head-stage-execution-20260920.md).

## Evaluation entrypoint

The following explicit adapter options expose the measured implementations.
**Integration VERIFIED** at `04163905f047169a159b3e366d679afc86392757`:
server CPU checks **59 passed, 3 CUDA-skipped in 14.12 s**; targeted CUDA checks
**8 passed in 20.73 s** (partly overlapping the CPU set), both exit 0.
The real-checkpoint adapter check ran **05:22:27–05:24:09 UTC**, GPU 4, exit 0.
Both arms passed seven complete action comparisons to their own uncached eager
reference, including repeated instructions, another image/state with the same
instruction, changed instructions and episode reset. All 14 calls preserved
the executed visual/action budgets, effective options and checkpoint identity.
Graph and text caches were released on close. This is an integration check;
its setup-inclusive request times are not a new timing benchmark.

The [adapter audit](evidence/prompt-cache-20260920/adapter-0416390-audit.json)
independently rechecked the saved action arrays. Report SHA256:
`d0f38ecf2d0df38d0b7f15fcdbfc00717e86d7ae30c79cd9e67d546a8b30a557`.
Run from the frozen model worktree with the same GPU/thread environment as above:

```bash
.venv/bin/python scripts/sparse/verify_prompt_cache_adapter.py \
  --inputs "$OUT/head-stage-calibration-20260920/inputs-gpu4/manifest.json" \
  --out-dir "$OUT/prompt-cache-20260920/adapter-0416390"
```

Existing live SR jobs keep their frozen options and outputs. A new matched
Dense/Sparse pilot and independent 500-identity Spatial configurations are
committed in action-eval at `aa5f5fa7815702ee9dd00aedca2a9e94d0ff9133`.
All four configurations passed server validation/doctor; the platform's **199
tests passed in 112.55 s**, and generated schema bytes match the committed schema.
The evaluator dependency files and existing environment were preserved.

The first CPU integration test invocation at `eaf5add` returned **1 failed,
58 passed, 3 CUDA-skipped** (4.87 s, exit 1). The failure was the new test
requiring an action-guidance counter from eager conditioned Dense, which does
not construct guidance or publish that counter. Its action comparison and
cache hit/miss checks had passed. The test now treats an absent counter as zero;
no model computation or measured result was changed. The failed log is retained
alongside the successful rerun.

Sparse `policy.options`:

```yaml
action_horizon: 32
denoising_steps: 10
rng_mode: fixed_per_predict
prompt_cache: {capacity: 8}
visual_cache:
  refresh_every: 5
  token_keep_ratio: 0.1
  action_guidance_weight: 1.0
  graph_dispatch: all_transformers
```

For the matched strengthened Dense control, use the same prompt cache and replace
only `visual_cache` with:

```yaml
visual_cache:
  refresh_every: 1
  conditioned_frame_reuse: true
  graph_dispatch: all_transformers
```

Effective options enter the policy fingerprint. Every episode reset clears the
instruction cache; visual/action state remains local to each request. Cache hits
and actual visual/action work are emitted with each prediction. The current
measurement meets the latency target but does **not** yet meet the complete
paired-SR acceptance criterion or establish a full SR/latency Pareto frontier.
The [new closed-loop record](prompt-cache-paired-20260920.md) contains the completed
15-pair pilot and the independently running full Spatial comparison.
