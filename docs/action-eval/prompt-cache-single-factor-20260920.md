# Exact prompt reuse on matched Dense and guided visual sparsity

Status: **PLANNED; no new timing or SR claim**. Date: **2026-09-20 UTC**.
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

Memoization has a bounded eight-entry LRU, exact strings/order/batch keys,
fresh tensor copies on return, and invalidation on encoder parameter/buffer,
device, dtype or tokenizer configuration changes. It requires frozen eval
weights. It never caches observations or actions. Cold instruction misses
are charged separately, and no miss is relabelled a warm hit.

`benchmark_prompt_cache.py` uses the three hash-verified real calibration
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
Reuse the existing Python 3.10.20 / torch 2.7.1+cu126 server environment without
installs or sync. Launch only on an admitted GPU and leave one available card
unused by this task. Code, admission, exact commands, exits and raw evidence
will be recorded after execution. The preceding compact Head/Stage negative
result remains [fully preserved](head-stage-execution-20260920.md).
