# Every-step 10% visual tokens without cross-step reuse

Status: **IMPLEMENTED; server verification and measurements pending**, 2026-09-20 UTC.
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

Require full-budget/native parity, an independent masked-reference check,
actual QKV/FFN/attention dimensions, unchanged action/scheduler counts, and
own-eager CUDA parity across changing observations before rollout. Reuse the
existing server environments without installation/sync. Every launch rechecks
resources and leaves another available card unused. Then use a new matched
50-identity exploratory cohort; errors are not task failures and incomplete
coverage retains null SR. No latency or SR result is claimed yet.
