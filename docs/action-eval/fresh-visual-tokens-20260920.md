# Every-step 10% visual tokens without cross-step reuse

Status at **2026-09-20 09:25 UTC: implementation/latency VERIFIED; renderer smoke FAILED; SR unavailable**.
User decision recorded **2026-09-20 11:31 UTC: preserve results; defer SR until an empty card is available** ("先保留结果，等空卡再测 SR"). No GPU job is launched in this update. The measured implementation and raw evidence remain unchanged.
The user resumed work on GPUs 4–7 and requested token sparsity at every step
instead of the prior visual-cache mechanism. Historical 500-episode queues stay
stopped. Current screening remains 50 episodes per candidate.

Each of the ten denoising steps, including step 0, selects 30/294 current visual
tokens and executes all 30 transformer layers on those rows. There is no dense
anchor and no historical visual K/V, hidden state or prediction. Action rows
are fully recomputed every step. Their attention accesses the selected current
visual keys plus all action keys; unlike the old cache, AV support also shrinks.
Unselected rows bypass the transformer using this step's pre-DiT input, then
the original full-grid video head and both unchanged schedulers run. This is
an approximation to video dynamics, not just an execution optimization.

Action-guided selection projects the current first-layer visual K for all 294
tokens and current action Q/K/V once per step. Joint A→[V,A] mass ranks tokens,
with ten slots for each of the three latent frames. The extra full-video K
probe is charged. The first action Q/K/V is used again within the same step's
first transformer layer; no tensor crosses a denoising-step boundary as a cache.
The equal-budget uniform control uses the same frame quotas and sparse math.

Visual transformer token-layer work is 9,000/88,200 (10.204%, ceil rounding),
plus 2,940 full-video K probe rows for the guided arm. Pre/post-DiT remains full
shape and all 300 action-layer updates remain. This is token-level computation;
no Head×Stage classification or FFN-neuron pruning is enabled. Frame quotas
provide structural coverage but do not establish the full AV–VV bridge.

Four timing arms share exact prompt memoization and transformer CUDA graphs:
full-recompute Dense, the stronger invariant-frame Dense, uniform fresh 10%,
and action-guided fresh 10%. Graphs rerun kernels on overwritten current inputs;
they do not reuse visual results. Full requests, current-input selection and
CPU action output are timed; graph construction and first prompt misses are
separate. Use three hash-verified real inputs and balanced rotating order.

## Complete-request latency and action diagnostics

Source **`6c52f362a975e560707b905c1c57cb46a1b9b0a8`**. Real-checkpoint timing ran
**09:11:30–09:12:42 UTC**, exit 0, policy GPU 4
(`GPU-490b4a76-6210-31b9-4e03-838a113cf5f4`), explicitly shared with CUDA
process 551992. GPU 6 was left unused by this task. Other applications were
active on the host. These are shared-load measurements, not an exclusive-GPU
claim; all samples and their substantial spread are retained.

| Variant | Mean ms | p50 ms | p95 ms | vs stronger Dense |
|---|---:|---:|---:|---:|
| Full-recompute Dense, same graphs/prompt optimization | 303.055 | 315.136 | 315.993 | — |
| Stronger Dense, also exact conditioned-frame reuse | 269.266 | 276.126 | 282.147 | 1.000× |
| Every-step uniform 10% | 222.261 | 232.378 | 239.422 | **1.211×** |
| Every-step action-guided 10% | 226.122 | 240.563 | 241.509 | **1.191×** |

Against full-recompute Dense, the guided arm is **1.340×**, also below 1.5×.
Action guidance adds **3.862 ms** to the equal-budget uniform mean. No part
of the previous cached method's 1.872× is attributed to this new method.
All ten visual layer sequences still execute and their weights are needed
at every step; a 10% token-row count is not a 10× wall-time prediction.

The 96 requests comprise 24 per arm. An independent audit confirms 48
variant/input/order-position cells with two observations each, reproduces
every mean and checks all executed counters. Both sparse arms execute 9,000
visual token-layer updates and 300 action-layer updates, select ten times,
and report zero reused visual steps. Every selected set has ten tokens per
frame. All 96 outputs and 20 additional graph checks matched their own
uncached eager actions bitwise; full and stronger Dense matched each other.
Neither parity statement means that Sparse matches Dense.

Action relative L2 versus Dense on the three exposed inputs:

| Input | Uniform 10% | Action-guided 10% |
|---|---:|---:|
| 0 | 0.9713 | 1.0002 |
| 1 | 1.0931 | 0.6341 |
| 2 | 1.0635 | 1.0649 |

These are large action differences, not an SR estimate. They prevent assuming
that the old cached candidate's pilot transfers to the fresh candidate. No
quality-preservation claim is made before the new closed loop.

## Verification, provenance, and closed-loop admission

Server CPU checks: **265 passed, 13 CUDA-skipped in 6.85 s**, exit 0. The targeted
CUDA-capable invocation: **26 passed in 8.59 s**, exit 0; it overlaps the CPU
suite and includes smaller wiring checks, rather than 26 entirely new tests.
Checks cover full-budget/native parity, an independently implemented full-shape
masked reference, physical QKV/FFN/attention dimensions, original visibility,
current-input identity bypass, complete action/scheduler counts, and graph
parity across changing observations and poisoned staging buffers.

Real-checkpoint adapter validation **09:12:51–09:13:39 UTC**, exit 0: seven
complete predictions for each of Dense and Sparse, including changing images
under a repeated instruction, changing instructions and episode reset.
All **14 calls** matched their own uncached eager references; executed options,
sampling counts, immutable model parameters and cache/graph cleanup passed.
The worker publishes `fresh_visual_tokens` separately from `visual_cache`.

Checkpoint SHA256:
`6c087e5b9e201f19dbe92834b470a7ee18937a18ea9adcbb49886aa1e3ac4c61`.
The existing Python 3.10.20 / torch 2.7.1+cu126 environment was reused without
installation, sync, precision or dependency changes. Horizon 32, replan 10,
ten denoising steps, RNG, released schedulers and protocol remain unchanged.

The frozen evaluator/configs are **`e0d9e80`**. Both new `dreamwam-fresh-*-quick50`
configs pass validate/doctor, all 199 platform tests pass with exit 0, and generated schema bytes
match. Initial doctor failed before rollout because the new worktree lacked
the external checkpoint link; adding the link to the unchanged checkpoint
resolved this data-mount omission. Original failure output is retained.
An initial exclusive GPU-4 admission also refused to launch when the existing
foreign CUDA job appeared; subsequent measurements explicitly declared sharing.

Closed-loop resources now have other applications' compute on 4/6 and graphics
on 5/7. The committed bounded launcher (`599d225`) can explicitly share an
existing graphics-only renderer, refusing any CUDA process there or a new
unapproved graphics process. It first runs a one-accepted-episode smoke check
on policy GPU 4 / renderer GPU 5. This is a new shared-rendering lineage;
native-read completion is checked, not assumed to prove pixel equivalence.
The smoke preserves its original 50-identity manifest and reports no SR.
Only after inspecting that result does a new independent 50-pair cohort run.
Every launch leaves another low-utilization compute-capable card unused by this
task. Errors remain separate from task failures; incomplete coverage retains
null SR. Old 500-episode plans and the prior paused quick50 plan are not resumed.

**Actual smoke outcome:** controller 571773 / runner 571779 reached the verified
fresh-token policy fingerprint, but the first native camera read recorded
`1 start` followed by `1 error unwritten_rgb`. The runner exited **73** before
any valid episode outcome. The original 50-identity manifest is preserved,
**accepted outcomes are zero and SR is null**; this is a renderer error, not
50 task failures or evidence about policy quality. The smoke audit confirms
the owned controller, runner and policy process have all exited. No full-pair
controller was started, and there is no automatic retry on the failed placement.

The system loader does not find `OSMesa`; no dependency was installed or changed.
MuJoCo documents OSMesa as a possible software renderer in its
[official rendering documentation](https://mujoco.readthedocs.io/en/3.3.1/programming/),
but that does not make it available in this preserved environment. The user chose
to preserve the results and wait for an empty card before testing SR. Once available,
recheck its occupancy, use the same card for both arms, rerun a bounded native
render check, then launch the existing new 50-pair cohort. Prefer an empty
renderer and omit `--allow-shared-graphics` for that launch; do not assume sharing
works because all model-only checks passed. The failed smoke never contributes
outcomes to the later cohort.

Exact commands, first-prompt misses, graph setup, all timing samples, action
arrays and the independent audit are in
[`evidence/fresh-visual-tokens-20260920/`](evidence/fresh-visual-tokens-20260920/).
Server root: `outputs/fresh-visual-tokens-20260920/`.
Timing manifest SHA256: `b5466001bddd2b842c9d1e58a6fdb974d9db030e350bffb5165a6c87555c4441`;
journal SHA256: `04a40187079d9d7dea3c7399590c86269d4898d8e4ace55d5c4f93e607c218e9`;
adapter report SHA256: `2cd0e2695403d285853628dbbbf52e660dd0016aa66a0836c61037eb56585283`.

```bash
# Frozen model checkout; GPU admission, environment, and paths recorded above.
.venv/bin/python scripts/sparse/benchmark_fresh_visual_tokens.py \
  --inputs "$REAL_INPUT_MANIFEST" --out-dir "$FRESH_FACTOR_OUT"
.venv/bin/python scripts/sparse/verify_prompt_cache_adapter.py \
  --fresh-visual-tokens --inputs "$REAL_INPUT_MANIFEST" --out-dir "$ADAPTER_OUT"
```
