# Temporal graph four-suite coverage and preserved Dense baselines

Status: **INCOMPLETE; no candidate benchmark SR**. This expansion uses the
existing frozen interval-5/all-token graph policy, without the new
[conditioned-frame factor](conditioned-frame-single-factor-20260920.md).
The acceptance plan requires Spatial/Object/Goal/Long, ten tasks × 50 states
each, or **2,000 episodes per configuration**. Spatial alone is insufficient.
The user confirmed on **2026-09-20 UTC**: deliver the complete Pareto results
first, then decide the acceptable SR decrease. The tolerance remains unset;
report paired results and latency without declaring non-inferiority or
retrospectively treating a later tolerance as preregistered. Previously inspected
official pilot states are not wholly held out.

## Baseline audit

A strict record audit on **2026-09-20 UTC** checks manifest identities,
status/success/termination agreement, single accepted terminal attempts and
record hashes. Original Dense runs have:

| Suite | Run suffix | Accepted / planned | Successes | Task failures |
|---|---|---:|---:|---:|
| Spatial | `run-20260919T004630Z-e33992a8` | 500 / 500 | 493 | 7 |
| Object | `run-20260919T021926Z-664b4c04` | 500 / 500 | 497 | 3 |
| Goal | `run-20260919T004630Z-ac289c8e` | 500 / 500 | 496 | 4 |
| Long original full plan | `run-20260919T005030Z-6b07c1b6` | 0 / 500 | 0 | 0 |
| Long historical partition B | `run-20260919T063434Z-4095670e` | 7 / 150 | 7 | 0 |
| Long historical partition C | `run-20260919T010352Z-90ebeaf9` | 17 / 150 | 15 | 2 |

The three completed short suites are preserved, not rerun for this expansion.
The Long partitions remain distinct incomplete runs; they are not a completed
500-identity baseline. Original short-suite provenance is clean model
**0ea3d7b** and evaluator **a19fe92**, the same Joint checkpoint SHA-256
`6c087e5b9e201f19dbe92834b470a7ee18937a18ea9adcbb49886aa1e3ac4c61`,
ten denoising steps, horizon 32, nine video frames, fixed-per-predict seed 42,
wait 30, replan 10 and 256×256 observations. Source review finds no change in
the LIBERO backend between the baseline and frozen new evaluator; episode-loop
changes optionally record input hashes and predicted/executed actions. Default model changes are opt-in wrappers and
diagnostics; prior native/r1 parity tests remain the numerical control.

## Frozen expansion

Configs are action-eval **334d28c**, in
`action-eval-temporal-suites-334d28c/configs/experiments/`:
`dreamwam-temporal-graph-libero-{object,goal,long}.yaml`.
All use model **2c02c5c** (`DreamWAM-graph-policy-2c02c5c`), evaluator
**4701ac2** (`action-eval-visual-4701ac2`), interval 5, token ratio 1.0,
literal guidance weight 1.0 and `graph_dispatch: all_transformers`. Full budget
does not construct action mass. Model/environment/precision/RNG are unchanged.
Object and Goal reference `dreamwam-release-v1` (max_steps 400); Long references
`dreamwam-release-v1-long` (max_steps 700). Both retain wait 30 / K 10 / seed 42.
The same policy's Spatial pilot completed 15/15; that pilot cannot fill any
full-run denominator.

Before launch, each actual GPU configuration passed `validate` and `doctor`,
exit 0; doctor does not load a model or render. These shared H200s had low
utilization and at least 98,776 MiB free at admission. GPU 3 was left unused.

| Suite | GPU | Independent run | Config fingerprint prefix |
|---|---:|---|---|
| Object | 5 | `object-gpu5/runs/run-20260920T011857Z-f8c5a29a` | `f8c5a29a` |
| Goal | 7 | `goal-gpu7/runs/run-20260920T011900Z-67e4152d` | `67e4152d` |
| Long | 6 | `long-gpu6/runs/run-20260920T013204Z-7ab8fda5` | `7ab8fda5` |

These paths are below
`dreamwam-sr/outputs/temporal-graph-suite-expansion-20260920`.
Preflights for all three on GPU 5 also exist as configuration checks; the actual
per-run preflight/fingerprint above is authoritative for GPU 7/6 launches.

```bash
# From action-eval-visual-4701ac2; frozen model/config paths described above.
export MODEL_ROOT LIBERO_ROOT GPU_UUID CONFIG
export CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU" OMP_NUM_THREADS=1
export MUJOCO_GL=egl PYTHONFAULTHANDLER=1 PYTHONPATH="$PWD/src:$LIBERO_ROOT"
unset MUJOCO_EGL_DEVICE_ID
.venv/bin/python -m action_eval.cli validate "$CONFIG"
.venv/bin/python -m action_eval.cli doctor "$CONFIG"
.venv/bin/python -m action_eval.cli run "$CONFIG" --output "$NEW_RUN_PARENT/runs"
```

The numeric parent CUDA mask keeps native rendering and the policy on the
declared physical device. No experimental EGL backend or alternate rendering
placement was enabled, and no foreign process was signalled.

## Errors, accepted failure and current coverage

Each expansion's initial runner exited **134** in native `read_pixels`.
Object had no initial terminal outcomes; Goal had three successes; Long had no
terminal outcome. Recovery helper **f8eae67**, SHA-256
`eb26dd9f276c5fa87bed6658b64c81cb040320f551eb99896e08b7af67e177c2`,
preserves the frozen manifest/config and every accepted success **and failure**.
Each batch admits at most 20 native-abort retries, waits for the previous policy
worker to exit, and checks utilization ≤20% / free memory ≥35,000 MiB. Initial
zero-progress Object/Long runs use a conservative two-consecutive-zero recovery
cap; Goal uses three. A productive batch can reset its internal zero-progress
counter; a stopped stagnation cap is not reset under unchanged conditions.

At **01:38:41 UTC**, the audited state is:

| Separate run | Terminal / planned | Successes | Task failures | Controller |
|---|---:|---:|---:|---|
| Matched Dense Spatial | 439 / 500 | 431 | 8 | batch 08, PID 255824, live |
| Eager temporal Spatial | 259 / 500 | 259 | 0 | batch 06, PID 251154, live |
| Graph temporal Spatial | 236 / 500 | 236 | 0 | batch 05, PID 255822, live |
| Graph temporal Object | 3 / 500 | 2 | 1 | stopped, stagnation cap |
| Graph temporal Goal | 3 / 500 | 3 | 0 | stopped, stagnation cap |
| Graph temporal Long | 0 / 500 | 0 | 0 | recovery 01, PID 255603, live |

Object recovery stopped after five invocations, the last two without progress;
Goal stopped after three zero-progress invocations. Both exit 1. Their caps are
respected. Object's accepted failure is **t000-i001-r00-s42**, hash
`5d7cd05a98a1b0f69b12710404a50256be7cb819b0db0b670af1195f145e5c34`;
the preserved Dense baseline succeeded on this identity. This is one official
discordant pair, not an overall rate or isolated causal explanation. The new
eighth Dense Spatial failure is **t008-i037-r00-s42**; it is retained along with
the previous seven. No settled task is retried to obtain a better outcome.

At **01:49:28 UTC**, Long's second recovery also exited with native SIGABRT
without a terminal outcome. Together with the initial launch, three invocations
made no progress. The helper stopped with `stagnation_limit`, exit 1; PID 255603
is absent. Long stays **0/500, zero successes and zero task failures**. Object,
Goal and Long are now all stopped at their respective caps, not restarted under
unchanged conditions. The refreshed strict audit has Dense Spatial **466/500
(458 successes, eight failures)**, eager temporal **277/500 successes** and
graph temporal **267/500 successes**. All three Spatial supervisors remain live:
255824 (Dense batch 08), 261798 (eager batch 07) and 255822 (graph batch 05).

The [evidence bundle](evidence/temporal-suite-expansion-20260920/) contains the
baseline audit with all accepted record hashes, protocol and policy fingerprints,
actual-GPU preflights, initial exit logs, bounded-recovery summaries and timestamped
progress snapshots. Live startup provenance may temporarily omit a policy
description while loading; earlier completed startup evidence remains authoritative.
Next: finish the productive Spatial plans and preserve all negative evidence.
A documented changed hardware condition is required before another bounded
attempt at the stopped suites. GPU assignment is part of the frozen config
fingerprint, so a hardware move requires a separately labelled run, rather
than editing `resolved.yaml` or pooling its outcomes with a different run.
Full Long Dense coverage and a valid four-suite paired comparison remain required.

## Matched Dense Spatial complete; Long moves to an independent GPU 3 run

At **2026-09-20 02:04:55 UTC**, matched Dense Spatial
`visual-cache-spatial-20260919/matched/run-20260919T220834Z-87c8f1d0`
has **500/500 terminal outcomes: 492 successes, eight official task failures,
SR 98.4%**. The evaluator summary is complete, with zero errors, cancelled or
unrun identities. Every CSV row agrees with its accepted result record and
frozen manifest; all accepted record hashes are retained. Recovery batch 08
ended with `complete_coverage` after 15 invocations, final runner exit 0 and
supervisor exit 0. The worker is absent. The policy remains **28845c6**,
`refresh_every: 1`, evaluator **4701ac2**, the checkpoint hash above and
`dreamwam-release-v1`. Provenance's untracked `.venv` / `pretrained` links are
retained in the evidence rather than relabelled as a clean checkout.

The full paired comparison with original Dense Spatial, executed on the server
using `scripts/sparse/paired_sr.py` at **e453bec**, also exits 0:

| Measure | Original Dense → matched refresh-1 Dense |
|---|---:|
| Terminal pairs / planned | 500 / 500 |
| SR | 98.6% → 98.4% |
| Absolute change | −0.2 percentage points |
| Original success → new failure | 6 |
| Original failure → new success | 5 |
| Unchanged outcomes | 489 |
| Task-stratified paired bootstrap 95% interval | [−1.4, +1.0] percentage points |
| Exact two-sided McNemar p | 1.0 |

The bootstrap uses 10,000 resamples and seed 42; its interval is conditional
on these benchmark tasks. This comparison does not prove equivalence or set
the user's SR tolerance. It records baseline repetition variability alongside
the earlier first-input camera-hash discrepancy. The two runs use the same
scientific protocol but different recorded model/evaluator revisions, so the
11 discordant pairs do not isolate a single source of variability. Episode
wall time is not full-request model latency.

```bash
# Server, DreamWAM-conditioned-guided-e453bec; existing environment unchanged.
.venv/bin/python scripts/sparse/paired_sr.py \
  --dense "$ORIGINAL_DENSE_SPATIAL" --sparse "$MATCHED_DENSE_SPATIAL" \
  --label 'native-Dense repetition (matched refresh1)' \
  --json "$SPATIAL_OUTPUT/dense-repetition-paired.json"
```

The [complete Dense evidence](evidence/temporal-suite-expansion-20260920/dense-complete/)
contains its manifest, 500-row CSV, evaluator summary, strict accepted-record
audit, public provenance and full paired statistics.

At **02:11:14 UTC**, eager temporal Spatial has **328/500 (327 successes,
one failure)**; graph temporal has **312/500 (310 successes, two failures)**.
Eager's failure is `t005-i033-r00-s42`; graph's are `t005-i028-r00-s42` and
`t006-i003-r00-s42`. Both Dense controls succeeded on the first and third;
original Dense failed on `t005-i028` while matched Dense succeeded. These
accepted outcomes and hashes are preserved in the paired-record evidence.
Neither candidate has complete SR. Graph batch 05 stopped at its 20-attempt
limit with recent progress. After the old controller and worker exited and
GPU 2 was verified empty, batch 06 resumed that same frozen run with the same
20-attempt / three-zero-progress limits. Eager batch 07 remains active.

Dense's completion released GPU 0. A fresh inventory confirmed GPUs 0 and 3
empty; **GPU 0 is now left unused**. At **02:05:49 UTC**, a new independently
labelled Long run starts on GPU 3:
`long-gpu3/runs/run-20260920T020549Z-1cb163e4`, fingerprint
`1cb163e41a00508f2d815941d9a9dc59a99879d4054aa08595eefeb6e26b390f`.
Its actual-GPU `validate` and `doctor` checks pass with exit 0. It retains
config **334d28c**, model **2c02c5c**, evaluator **4701ac2** and the official
500-identity Long / max-700 protocol. Native renderer and policy both use
physical GPU 3 (`GPU-9aa453a3-63f9-ea74-f4ce-7d119e9e69ba`). At the audit,
runner 277918 and policy 277942 are live with no terminal outcomes yet.
The stopped GPU 6 run remains unchanged and its outcomes are not pooled.
No renderer workaround, precision, weights, RNG or protocol change is enabled.
