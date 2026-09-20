# Native refresh, budget and structure follow-up

Status: 380 bounded predictions completed and independently audited on H100,
2026-09-20 UTC. Cumulative checkpoint predictions: **847**; closed-loop attempts:
**0/50**. These are two exposed DreamWAM development observations, not DIDO author
records or SR evidence. Exact revisions, timestamps, commands, failures and 51
artifact hashes: [FOLLOWUP-VERIFICATION.json](FOLLOWUP-VERIFICATION.json).
The prior [20-candidate screen](ONLINE-RESULTS.md) remains separate.

Each row has four warm timings (two repeats on two unique inputs), interleaved
with stronger Dense and historical uniform feature reuse. Timings cover full
prediction including online scoring/packing. Error is mean/worst relative L2 of
the first ten pre-binarization actions. Shared-host ratios from different stages
are not a direct latency comparison.

| Shared context R56/U30 schedule | Warm speedup | Mean raw prefix L2 | Worst | Grip |
|---|---:|---:|---:|---|
| D0, no Sparse refresh | 2.353× | 0.109831 | 0.112862 | 0 / 0 |
| S1 | 2.043× | 0.139701 | 0.164819 | 0 / 0 |
| S2 | 1.999× | 0.142744 | 0.164815 | 0 / 0 |
| S3 | 1.958× | 0.143944 | 0.167157 | 0 / 0 |
| S4 | 1.859× | 0.138186 | 0.168848 | 0 / 0 |
| S5 | 1.962× | 0.145320 | 0.169962 | 0 / 0 |
| S6 | 1.946× | 0.142867 | 0.164277 | 0 / 0 |
| S7 | 1.916× | 0.307261 | 0.440318 | 1 / 0 |
| S8 | 1.932× | 0.162567 | 0.163789 | 0 / 0 |
| S9 | 1.870× | 0.124374 | 0.128103 | 0 / 0 |

Every single Sparse refresh worsens mean error here. Measured S9/S4/S1 ranks
determine the two multi-refresh controls below. Dense-anchor native ranks remain
aged; Sparse steps recompute current drift-selected patch features without
secretly refreshing all-depth native rankings.

| Budget/multi-refresh candidate | Warm speedup | Mean raw prefix L2 | Worst |
|---|---:|---:|---:|
| No refresh, R28 | 2.184× | 0.170818 | 0.179585 |
| No refresh, R84 | 2.164× | 0.105526 | 0.105903 |
| S9, R28/U15 | 1.963× | 0.239329 | 0.305826 |
| S9, R56/U15 | 1.902× | 0.125520 | 0.140947 |
| S9, R84/U15 | 1.882× | 0.110593 | 0.111929 |
| S9, R56/U45 | 1.977× | 0.206422 | 0.245525 |
| S4 + S9, R56/U30 | 1.808× | 0.144223 | 0.165571 |
| S1 + S4 + S9, R56/U30 | 1.633× | 0.143747 | 0.167237 |

All eight rows have zero prefix gripper disagreements. R28 gives little useful
speed gain while increasing error. R84 slightly improves the no-refresh proxy;
retain it alongside shared R56 and layerwise value-aware R56 for the nine-input
comparison. U45 at S9 is substantially worse than U15/U30; neither multi-refresh
schedule beats no refresh. No adaptive refresh threshold is adopted.

| Structure reuse with fresh retained Q/K/V every step | Warm speedup | Mean raw prefix L2 | Worst | Grip |
|---|---:|---:|---:|---|
| Uniform R56 | 1.255× | 0.816225 | 1.102295 | 5 / 0 |
| Shared native context R56 | 1.214× | 0.779987 | 1.288630 | 5 / 0 |

Both structure controls recompute 56 retained rows at each step after D0 and
reuse no old visual K/V. Neither reaches 1.5×; both substantially change actions.
The roughly 2× feature-mode gain cannot be attributed to fresh-every-step sparsity.
Context improves mean but worsens worst-case error versus uniform here.

The budget screen archives actual hard-read routes and Sparse query indices in
24 untimed eager reference calls, covering 240 steps. Independent replay verifies
route hashes, uniqueness, frame/depth coverage, unchanged Reuse structure and
`R_new - R_old ⊆ U ⊆ R_new`. Graph/timed outputs match their own trace-enabled
eager references byte for byte; timed calls use counter diagnostics. At shared
Dense anchors the historical `query` field denotes the route seed; `q_rows=294`
records the actual full Dense work. All 380 raw archives and 160 new warm rows
are audited. CPU builder/archive checks pass 7, then 8, then 9 tests at successive
revisions; model runtime code and earlier actual CUDA gates are unchanged.

Two initial refresh launches were refused before loading and made zero calls.
The bounded admission wait retains ≥50,000 MiB free / ≤10% utilization, records
every refusal and waits at most 120 seconds. It retries no prediction.

CPU OSMesa at evaluator `5eab0bd` passes validate/doctor, action/observation contract
description and 80/80 native RGB reads. Its five zero-action diagnostic steps
load no policy, initialize no CUDA and yield no SR. Failed CLI-entrypoint and
LIBERO-import-path attempts are preserved; invocation/environment paths were
corrected without installing dependencies.

Reproduce refresh/structure with source `6c9a8bd`, budget follow-up with `dca3413`,
and trace audit with `97a8596`. Use `scripts/sparse/benchmark_native_routes.py`
with `--stage refresh`, `--stage structure`, or `--stage frozen --design-key
online_budget_followup`; GPU invocations add `--share-gpu5 --admission-wait-seconds
120 --out-dir <fresh-directory>`. Exact historical commands/artifact hashes are
in the linked record; do not duplicate completed finite runs automatically.

Next: the predeclared 277-call nine-input comparison, final adapter/reset checks,
then the six-attempt paired development pilot. The five-point SR margin remains
untested. Weak empirical correlation is handled by rejecting development
candidates; there is no calibrated online semantic-correlation detector. Runtime
uniform fallback covers invalid numerical scores, with no timed fallback events.
