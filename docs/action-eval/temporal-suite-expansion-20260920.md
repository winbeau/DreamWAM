# Temporal graph four-suite coverage and preserved Dense baselines

Status: **INCOMPLETE; no candidate benchmark SR**. This expansion uses the
existing frozen interval-5/all-token graph policy, without the new
[conditioned-frame factor](conditioned-frame-single-factor-20260920.md).
The acceptance plan requires Spatial/Object/Goal/Long, ten tasks × 50 states
each, or **2,000 episodes per configuration**. Spatial alone is insufficient.
The SR tolerance is still unspecified; report exploratory paired results and
Pareto tradeoffs without declaring non-inferiority or retrospectively choosing
a tolerance. Previously inspected official pilot states are not wholly held out.

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

The [evidence bundle](evidence/temporal-suite-expansion-20260920/) contains the
baseline audit with all accepted record hashes, protocol and policy fingerprints,
actual-GPU preflights, initial exit logs, bounded-recovery summaries and timestamped
progress snapshots. Live startup provenance may temporarily omit a policy
description while loading; earlier completed startup evidence remains authoritative.
Next: finish the productive Spatial plans, preserve all negative evidence,
continue Long only within its current bounds, and use a documented changed
hardware condition before retrying stopped Object/Goal runs. Full Long Dense
coverage and a valid four-suite paired comparison remain required.
