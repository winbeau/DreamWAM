# Visual refresh interval 5 versus 10 with a strengthened Dense control

Status: **MEASURED; no interval-10 official SR**. The independent confirmation
on **2026-09-20 02:25:00–02:28:01 UTC**, exit 0, measures full warmed requests
at **265.09 ms Dense, 148.98 ms interval 5 and 134.36 ms interval 10**.
The ratios against equally optimized Dense are **1.779318×** and
**1.973067×**. Changing only the visual cadence adds **1.108889×** speedup.
These are synthetic-input latency controls; the interval-10 row is neither
an accepted SR candidate nor the paper's action-guided M1–M3 method.

## Fixed experiment and verification

Benchmark commit **053e25553dcb6b1de070c4e61a7fae0fbab0b3e6**, pinned server
worktree `DreamWAM-cadence-053e255`. The only new application file is
`scripts/sparse/benchmark_visual_cadence.py`; it invokes existing cache classes.
Local commit/push preceded a clean server pull and detached worktree. Server
syntax compilation and both real-checkpoint measurements exit 0. No dependency
installation or sync occurred; Python 3.10.20 / torch 2.7.1+cu126 are unchanged.

The Joint bf16 checkpoint SHA-256 remains
`6c087e5b9e201f19dbe92834b470a7ee18937a18ea9adcbb49886aa1e3ac4c61`.
Horizon 32, nine video frames, ten denoising steps, seed 42, CPU RNG and the
released schedulers are unchanged. `fast_ops_enabled` is false. The script
uses synthetic camera seeds 0/1/2, zero proprioception and a fixed instruction;
correctness-only requests also change language/proprioception and return to
the first input. It loads the real checkpoint, not surrogate layer weights.

Conditioned-frame reuse and transformer graph scope are fixed in both cadence
variants and in the applicable Dense control. Interval 5 has one full visual
anchor, one refresh of all future-frame rows and eight visual-reuse steps.
Interval 10 has one full anchor and nine visual-reuse steps. Both execute all
**300 action-layer updates**, ten action steps and both schedulers. There is
no AV guidance, sparse VV-key route, new head classification or partial
future-token selection in this cadence factor.

Every run measures seven variants × 42 repetitions = **294 full requests**.
Rotating order balances all seven positions and all three inputs twice for
each variant. Two warmups per variant and three graph-capture warmups precede
measurement. Timing spans synchronized `predict_action` through CPU actions;
model load and first-call/capture costs are recorded separately.

Both runs retain every sample. Across **588 timed requests** and **70
correctness-only variant requests**, all graph outputs are bitwise their own
eager reference. The conditioned variants also match their corresponding
pre-conditioned-reuse reference actions. Independent audits reproduce the means
and sample balance. Computation counters, request isolation and unchanged model
parameter versions/evaluation settings are checked. This does not mean that
temporal actions match Dense or that synthetic parity establishes SR.

## Complete measurements, including the shared-load run

| Full request | GPU 1 first run mean ms | GPU 0 confirmation mean ms | Confirmation p95 ms |
|---|---:|---:|---:|
| Native Dense eager | 476.25 | 461.08 | 461.80 |
| Conditioned-frame Dense eager | 468.62 | 461.81 | 462.33 |
| Interval 5 eager | 286.54 | 280.88 | 280.61 |
| Interval 10 eager | 260.74 | 258.27 | 257.53 |
| Conditioned-frame Dense graphs | 260.25 | **265.09** | 269.36 |
| Interval 5 graphs | 146.92 | **148.98** | 151.79 |
| Interval 10 graphs | 135.38 | **134.36** | 136.93 |

First run: **02:19:38–02:22:26 UTC**, GPU 1, exit 0. The card was empty at
admission, but foreign CUDA PID 298176 subsequently appeared with 46,004 MiB.
The snapshot at **02:21:21 UTC** records 100% utilization and both that process
and benchmark PID 297535. It was not signalled. Large latency excursions
remain in the full 294-sample record; the run is explicitly shared-load.
It gives interval-5 / interval-10 ratios **1.771435× / 1.922365×** against
Dense, and a cadence-only factor **1.085202×**. No quiet sub-window is selected.

The independent confirmation uses GPU 0
(`GPU-0b850a78-f1db-6bda-8dd5-d96296bc1bfc`), keeping GPU 3 unused during the
measurement. Two-second process telemetry observes only benchmark PID 304456
on GPU 0; other GPUs remain shared. This is a sampled occupancy observation,
not an exclusive reservation. Early/late timing modes and the first-repeat
eager outliers remain in the means; medians, p95 and standard deviations are
in the raw summary. No repeated run was selected because of a desired ratio.

On the three timed inputs, interval-5 action relative L2 from Dense ranges
**0.09364–0.10035**; interval 10 ranges **0.12248–0.13164**. These diagnostics
are larger at interval 10, but neither quantify SR nor determine admissibility.
The user's SR tolerance remains unset pending complete Pareto results.

Confirmation loading takes **16.61 s**. First graph calls take **6.33 s Dense,
32.32 s interval 5 and 4.25 s interval 10**, versus 0.80 / 1.76 / 0.58 s in
the first run. They include graph setup and occur sequentially, so they are
not comparable lifecycle-startup estimates and are not concealed inside the
warmed speedup. Peak allocated/reserved memory is **25,205 / 25,632 MiB**.

```bash
# Frozen server worktree; existing environment, no install/sync.
PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/sparse/benchmark_visual_cadence.py \
  --reps 42 --out-dir "$NEW_CONFIRMATION_DIR"
```

The first run changes only physical GPU/output directory. Server artifacts:
`dreamwam-sr/outputs/visual-cadence-factor-20260920/`.
The [evidence bundle](evidence/visual-cadence-20260920/) preserves both manifests,
all 588 samples, summary/audits, load observation, telemetry and exit codes.
Ongoing official SR policies are unchanged. Next: complete paired quality
coverage and the [paper-mechanism evidence](paper-mechanism-status-20260920.md),
without presenting cadence-only speed as evidence for AV–VV bridging.
