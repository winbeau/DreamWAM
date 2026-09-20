# Native routing: bounded checkpoint screen

Status: three predeclared stages complete and independently audited on H100,
2026-09-20 UTC. This is development input replay, **not closed-loop SR**.
The source is `efd6c478fb63fafba6f1c19ed090c885fb7485b9`; the independent auditor
is `dea89198b8fdfb598052825d9970b538742fc380`. All checkpoints, dependencies,
sampling steps, input preprocessing and action horizons remain unchanged.

## Cohort and validation

Two already exposed observations are used: `t000-i001-r00-s42-c0004` and
`t001-i001-r00-s42-c0006`. Each candidate has two warm timing repetitions per
observation. Every two-candidate group includes contemporaneous stronger Dense
and the exact historical D0/R1–9, KV18.75 (`56/294`) uniform feature-reuse
control. Uniform is the anchor's effective behavior; the old config names its
inactive later-step selector `drift`.

The three stages execute **219 + 76 + 105 = 400 complete predictions**, including
all 84 eager references, 84 capture/changed-input calls, 42 warm-graph prompt
misses, 22 prompt-hit restoration calls and 168 warm timed calls. There are 400
hashed raw/executable action archives, totaling 716,800 uncompressed array bytes.
The first layer-stage invocation was refused before model load and made zero
calls; its immutable error directory is retained separately.

Every graph/setup call matches its own eager reference. The auditor additionally
requires **identical array bytes**, verifies the native-uniform arm against the
historical control, replays all action metrics and checks the actual native
token/layer budgets. All 400 archives and all 168 timing rows pass. No native
numeric fallback occurs in the 80 timed candidate calls. CUDA tests also pass
all four BF16 shared/layerwise/pool/structure scenarios after the recorded
host-index, fixture-device and additive-bias dtype corrections.

All methods here use D0/R1–9 **feature reuse**: 294 visual queries at D0 and zero
visual queries at steps 1–9. All 32 action tokens execute all 30 layers for all
10 denoising steps. The nominal recompute ratio 0.1 is inactive in this screen.
Scoring, selection, packing/pooling, request-local transfers and final model
output are included in complete `predict_action` timing. Archival copies and
hashes follow the timed call. First misses/capture are retained in `calls.jsonl`
and must not be described as warm measurements.

## Complete candidate matrix

Errors are relative L2 against matched stronger Dense. `raw prefix` means the
pre-binarization first 10 actions, averaged over the **two unique inputs**;
timing repetitions are not extra action-quality samples. `grip` is the number
of executable prefix gripper disagreements for the two inputs, not task failures.
Speedup is the ratio of paired mean complete-prediction latencies, with four
warm samples per candidate. These are descriptive small-sample points.

| Candidate | Read rows | Warm speedup | Full executable L2 | Mean raw prefix L2 | Worst raw prefix L2 | Grip |
|---|---:|---:|---:|---:|---:|---|
| Native uniform | 56 | 2.119× | 0.21187 | 0.32220 | 0.47028 | 1 / 0 |
| A→V | 56 | 2.130× | 0.09027 | 0.11482 | 0.13399 | 0 / 0 |
| V norm | 56 | 2.127× | 0.08385 | 0.11805 | 0.13478 | 0 / 0 |
| Value-aware A→V | 56 | 2.118× | 0.09157 | 0.11937 | 0.15346 | 0 / 0 |
| V video-time dynamic | 56 | 2.123× | 0.17652 | 0.28434 | 0.46783 | 1 / 0 |
| VV visual-context proxy | 56 | 2.123× | 0.17974 | 0.29098 | 0.46598 | 1 / 0 |
| AV-seeded VV context | 56 | 2.130× | 0.08790 | 0.10983 | 0.11286 | 0 / 0 |
| Value-aware, uniform observed frame | 56 | 2.125× | 0.09000 | 0.11286 | 0.11308 | 0 / 0 |
| Value-aware + 0.25 dynamic | 56 | 2.102× | 0.08409 | 0.11016 | 0.11376 | 0 / 0 |
| Value-aware + 0.50 dynamic | 56 | 2.126× | 0.17675 | 0.28405 | 0.47191 | 1 / 0 |
| Value-aware + 0.75 dynamic | 56 | 2.088× | 0.17901 | 0.28341 | 0.47090 | 1 / 0 |
| Layerwise value-aware | 56/layer | 2.039× | 0.06857 | 0.09676 | 0.10109 | 0 / 0 |
| Layerwise dynamic | 56/layer | 2.064× | 0.18209 | 0.28852 | 0.47079 | 1 / 0 |
| Layerwise AV-seeded VV context | 56/layer | 2.028× | 0.19865 | 0.30779 | 0.46081 | 1 / 0 |
| First-layer-only value-aware | 56 | 2.134× | 0.42202 | 0.33021 | 0.48550 | 1 / 0 |
| Hard uniform, full observed frame | 198 | 2.044× | 0.19849 | 0.29880 | 0.46465 | 1 / 0 |
| Pooled uniform, count multiplicity | 198 | 1.868× | 0.18358 | 0.29775 | 0.46431 | 1 / 0 |
| Hard dynamic, full observed frame | 198 | 2.079× | 0.17857 | 0.27908 | 0.46338 | 1 / 0 |
| Pooled dynamic, count multiplicity | 198 | 1.797× | 0.17999 | 0.28222 | 0.45905 | 1 / 0 |
| Pooled dynamic, unit multiplicity | 198 | 1.813× | 0.17585 | 0.27405 | 0.46077 | 1 / 0 |

## Findings and decisions

Multi-depth action/value/context proxies improve these two replayed action
vectors over uniform at the same 56-row budget. The strongest shared-route
prefix result is AV-seeded VV context; layerwise value-aware has still lower
mean/worst prefix error, at approximately 5% more warm latency than its matched
historical uniform control. Keep both as development candidates. Their speed
comes mainly from cross-step **visual feature reuse**, not faster scoring or
fresh-every-step recomputation. Both remain around the old control's speed.

The single-first-layer value-aware ablation is poor here. This isolates the
depth selection at the same D0 anchor; it does not by itself prove that early
denoising time is intrinsically unsuitable. Layerwise routing is not uniformly
better: layerwise context loses the improvement of shared multi-depth context.
These results support explicit depth/routing ablation rather than transferring
a blanket importance claim from DIDO.

The 0.25 dynamic mixture does not improve the shared context candidate's mean
or worst prefix error, while larger dynamic weights retain the same prefix
gripper disagreement as uniform on the first input. No fusion weight is adopted
by default. The pure endpoints and observed-frame treatment are recorded, so
the mixture comparison does not silently change that frame's selection rule.

Background pooling at 198 packed rows adds online cost. Count-weighted dynamic
pooling takes 145.81 ms versus 126.24 ms for equal-budget hard dynamic selection
and has slightly worse mean full-action and raw-prefix error. Unit multiplicity
slightly reduces mean error, but retains the first input's gripper disagreement
and is slower than hard selection. Full observed-frame coverage and pooled
backgrounds therefore do not solve this feature-reuse candidate's control proxy
problem. This is a result for this D0/R1–9 integration, not a general rejection
of DIDO's trained model or per-step refinement.

No closed-loop episode was executed. The remaining work includes bounded
Sparse refresh and budget scans, structure controls, actual selected-route
examples, stronger same-input timing on the retained choices, real adapter
reset/fingerprint checks, then the predeclared paired pilot. The 5-point SR
margin is not established by these action-vector results.

## Reproduction and artifacts

Use the immutable H100 worktree `.trees/dido-run-efd6c47-native`, unchanged
model interpreter and the workflow's explicit GPU/environment overrides:

```bash
python scripts/sparse/benchmark_native_routes.py --stage selectors --share-gpu5 --out-dir <new-selectors-directory>
python scripts/sparse/benchmark_native_routes.py --stage layers --share-gpu5 --out-dir <new-layers-directory>
python scripts/sparse/benchmark_native_routes.py --stage pooling --share-gpu5 --out-dir <new-pooling-directory>
```

These are historical reproduction commands, not authorization to duplicate the
completed finite screen. Before any later run, check the remaining effort and
resource budget. Source raw/report directories under H100
`outputs/dido-sparse-profile-20260920/` are:

- `native-selectors-efd6c47/`: 219 calls, 19:51:58–19:53:29 UTC.
- `native-layers-efd6c47/`: admission refusal, zero calls, 19:54:21–19:54:32 UTC.
- `native-layers-efd6c47-retry1/`: 76 calls, 20:01:33–20:02:19 UTC.
- `native-pooling-efd6c47/`: 105 calls, 20:04:14–20:05:12 UTC.
- `native-audit-dea8919/all-stages/`: complete independent report and all 168
  per-input timing/metric rows. Report SHA-256
  `94f9ab7ef5db917bc14aa55474a0bdeb2710703c8580a51e40d049430856081f`;
  CSV SHA-256 `9869bd6c11e509b6d1d4ec741ba4951bf69e0f9bfc524893302198295ae92fe4`.

Run the auditor at `dea8919` with one `--screen <directory>` for each completed
stage and `--out-dir <new-audit-directory>`. It rejects unfinished screens,
changed archives, incomplete setup/timing coverage, altered action budgets or
metrics, and output disagreement with archived own-eager controls. Complete
commands, hashes, CUDA failures/corrections and memory peaks are recorded in
[ONLINE-VERIFICATION.json](ONLINE-VERIFICATION.json).
