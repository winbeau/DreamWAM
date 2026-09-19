# Joint FFN cache: controlled full-request measurements

Status: **MEASURED / ongoing**. Date: 2026-09-19, UTC (local workspace: America/Los_Angeles).

The active objective is SR-constrained acceleration of DreamWAM Joint, at least
1.5× and aiming for 2×. None of the measurements below establishes that objective.
The current user authorization permits idle or lightly occupied GPUs. It supersedes
older GPU-pause and GPU-7-only notes; other users' processes remain untouched.

## What is actually measured

- Published checkpoint SHA256:
  `6c087e5b9e201f19dbe92834b470a7ee18937a18ea9adcbb49886aa1e3ac4c61`.
- Joint, bf16, 10 denoising steps, horizon 32, replan 10, 9 video frames,
  model seed 42, separate original CPU RNG generators, original world residual.
- Entire `predict_action`: text encoder, VAE, ten sampling steps, selection,
  cache creation/reconstruction, normalization and CPU action output. Synchronized
  wall time; no timing hooks in model operations.
- Native Dense and a 100% recomputation wrapper are both measured. Each candidate
  is compared with matched Dense in the same process/GPU/checkpoint, on the same
  inputs, with rotating order. The full-budget wrapper must match native actions
  bitwise for all inputs and on every measured Dense call.
- Three deterministic synthetic camera pairs with zero proprioception are used
  for latency screening. This is **not an official episode evaluation**. SR is null;
  action differences are diagnostics and cannot establish preservation of SR.
- Python 3.10.20, torch 2.7.1+cu126, existing server environment. No dependency,
  `pyproject.toml`, precision, or lock change. The model checkout has no `uv.lock`;
  no dependency synchronization was performed or claimed.

The full neural model and its checkpoint are used. In contrast, the earlier
`benchmark_token_compression.py::make_blocks` fills surrogate block weights with
0.02. Its results are useful as a microbenchmark, but must not be called a
real-checkpoint full-request test or a proof that every token/FFN strategy fails.

## Factor C1: relative-input-drift visual token recomputation

Implementation: [`ffn_context_cache.py`](../../dreamwam/sparse/ffn_context_cache.py).
First step computes each visual FFN densely. Later steps rank tokens by
`RMS(x - x_ref) / max(RMS(x), 1e-6)` and recompute the selected rows. Unselected rows
retain their last computed output. **Only selected input references advance**,
so accumulated drift on reused rows is not erased. All references are discarded
on normal and exceptional request exit. Action FFNs and all attention stay dense.

This transfers the Context Cache scoring rule from
[rMuscle §4.1](https://arxiv.org/html/2609.19104v1#S4.SS1) to adjacent denoising
steps of the Joint visual expert. It does not implement rMuscle's cross-execution
reference library, and has no action-guidance signal enabled in this ablation.

Run `keep010`: commit `7ea21a3e45dd9bcd7d62100755307b58821cfb17`, clean tree,
H200 NVL GPU 7 (`GPU-5e07f671-a7e1-b38d-4d14-30316dfd9ce8`), 2026-09-19
20:35:38–20:36:33 UTC, exit 0, two warmups and 15 measurements per variant.

| Variant | Mean ms | p50 ms | p95 ms | Speedup vs matched mean |
|---|---:|---:|---:|---:|
| Native Dense | 463.04 | 461.55 | 473.28 | — |
| Matched Dense, 100% rows | 466.45 | 464.57 | 475.61 | 1.000× |
| 10% rows after first-step anchor | 520.99 | 517.74 | 533.14 | **0.895×** |

**Negative result:** this implementation is 11.7% slower. Executed work is 30 dense
and 270 selective visual FFNs per request; 30/294 rows are selected on each sparse
call. Including the dense first step, it computes 16,920/88,200 token-rows (19.18%).
It does not mean only 10% of total request FFN work was executed. Full-budget
actions are bitwise equal; model parameter versions are unchanged. Maximum action
relative L2 on these synthetic inputs is 0.05781, not an SR estimate.

Small raw artifacts:
[manifest](evidence/context-cache-20260919/keep010-manifest.json),
[all requests](evidence/context-cache-20260919/keep010-requests.jsonl),
[summary](evidence/context-cache-20260919/keep010-summary.json).
Server artifacts are under
`dreamwam-sr/DreamWAM/outputs/context-cache-20260919/`.

Exact command, from the server model checkout:

```bash
CUDA_VISIBLE_DEVICES=7 .venv/bin/python scripts/sparse/benchmark_ffn_context_cache.py \
  --keep-ratio 0.1 --warmup 2 --reps 15 \
  --out-dir outputs/context-cache-20260919/keep010
```

Run `keep000` first attempt: exit 0 and full-budget parity passed, but the Dense
latency varied between approximately 463 and 672 ms. Mean speedup was 1.029×;
p50 ratio was 1.173×. This disagreement and the large within-run variation prevent
using that attempt as a stable estimate of the zero-recomputation boundary.
Raw records are retained; a fixed-revision repeat on idle GPU 6 is required.

The fixed-revision repeat completed at **20:44:04 UTC**, exit 0, on H200 NVL GPU 6
(`GPU-8cf0627c-5919-bace-1c42-e5d627c36cb5`). Commit `a1e59ab` was checked out
as a separate detached Git worktree with the existing model environment and weights;
`PYTHONPATH` selected that worktree. All variants used `OMP_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`, and `OPENBLAS_NUM_THREADS=1`. There were two warmups and
24 measurements per variant; GPU telemetry was captured once per second.

| Variant | Mean ms | p50 ms | p95 ms | Speedup vs matched mean |
|---|---:|---:|---:|---:|
| Native Dense | 453.23 | 452.93 | 456.93 | — |
| Matched Dense | 455.42 | 455.18 | 458.21 | 1.000× |
| First step dense, remaining visual FFNs fully reused | 447.47 | 447.75 | 451.25 | **1.018×** |

Every candidate request ran 30 dense and 270 fully reused visual FFNs, computing
8,820/88,200 rows (10% including anchors). Full-budget parity and unchanged parameter
versions passed. This is a measured boundary of this FFN-only implementation, not a
universal bound on broader visual reuse. It falls far short of the 1.5× objective.

Evidence: [manifest](evidence/context-cache-20260919/keep000-stable-manifest.json),
[requests](evidence/context-cache-20260919/keep000-stable-requests.jsonl),
[summary](evidence/context-cache-20260919/keep000-stable-summary.json),
[GPU telemetry](evidence/context-cache-20260919/keep000-stable-gpu.csv).
The noisy first attempt is preserved as `keep000-noisy-*` in the same directory.

```bash
PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=6 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python scripts/sparse/benchmark_ffn_context_cache.py \
  --keep-ratio 0 --warmup 2 --reps 24 \
  --out-dir outputs/ffn-20260919/keep000
```

## Factor N1: compact selected-neuron updates

Implementation: [`ffn_neuron_cache.py`](../../dreamwam/sparse/ffn_neuron_cache.py),
commit `a1e59ab`. This factor is measured with token recomputation disabled.
At a dense anchor, rank visual FFN neurons by `(sum |h|) * ||W_down[:, n]||_2`.
Gather the selected up-projection rows and down-projection columns once for the
group, then execute only these compact weights at later steps. Reconstruct
`y_anchor + (h_selected - h_anchor_selected) W_down_selected`. Unselected neurons
retain their anchor contribution; checkpoint weights are not pruned or modified.
Static weight norms are computed once at setup and timed separately; online mask
selection, weight gathering, dense anchors and reconstruction remain in requests.

The initial schedule uses one group of 10 steps and a 10% neuron budget. This is
an online anchor approximation to the Action Cache rule, transferred to visual
FFNs; it does not claim rMuscle's offline reference masks or cross-episode lookup.
No token cache, attention restriction or public operator optimization is enabled.

Server correctness checks at the fixed revision: **15 passed**, exit 0. They cover
actual FFN wrapper execution, selective rows, accumulated drift, request/error
isolation, dense parity, selected neuron delta reconstruction, compact weight sizes,
group boundaries, and unchanged action FFN entry points.

## Remaining work

Finish and archive the N1 full-request comparison. If FFN-only reuse is insufficient,
proceed to one separately measured
factor that removes broader visual token computation, accounting for projection,
attention, FFN and reuse costs. Add action guidance only as its own ablation.
Any candidate promoted to quality evaluation needs complete official episode
pairing; errors remain errors and incomplete coverage produces no SR. No result
here supports changing denoising steps, horizon, replan, RNG, weights, or protocol.
