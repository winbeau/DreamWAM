# Conditioned-frame reuse and a stronger common Dense control

Status: **MEASURED; official SR acceptance pending**. On **2026-09-20
01:35:09–01:37:24 UTC**, an empty-at-launch H200 measured full requests at
**254.80 ms** for Dense with transformer graphs and conditioned-frame reuse,
**142.87 ms** for temporal reuse with the same optimization (**1.783456×**),
and **140.49 ms** for the unchanged guided 10% candidate (**1.813595×**).
Guidance's low token budget is only **1.016900×**, or 2.37 ms, faster than the
strengthened temporal candidate here. These are latency results, not SR results.

The new common optimization also benefits Dense. The earlier approximately
1.97× temporal and 2.05× guided ratios used a Dense control without it. Those
measurements remain valid for their stated control, but cannot establish 2×
against this stronger control. The new factor remains benchmark-only; ongoing
official rollouts retain their original frozen policy code and options.

## Invariant audit before implementation

Diagnostic commit **e9cd379** observes native Dense without replacing any
calculation. The sampler restores the first video latent frame after every
step, keeps flow fixed, and sets its time to zero. Temporal patch size is one.
The attention mask prevents its queries from reading future frames or actions;
cross-attention depends on fixed request context, and world routing acts per
video token. Induction over layers therefore predicts a fixed conditioned frame.

The actual checkpoint audit completed **01:23:11–01:23:45 UTC**, exit 0. All
**98 of 294** conditioned video tokens had bitwise-identical Q/K/V, layer inputs,
post-attention/FFN outputs and post-world outputs across all 30 layers and all
ten steps. Five requests cover three different synthetic image pairs, a changed
language/proprioception request, and return to the first input. All **8,370**
conditioned/context comparisons were identical; all **8,280** future-frame
comparisons changed. Instrumented full actions matched separate uninstrumented
native calls exactly. This proves the observed invariant on these inputs, not
closed-loop quality. Reduced matrix shapes could still change floating-point
rounding, so subsequent action parity is measured separately.

The first diagnostic launch exited 1 before model loading because the new
worktree lacked its checkpoint data link. That failure is retained. Adding the
existing checkpoint/pretrained links fixed deployment; no dependency changed.

## One factor and fixed conditions

Implementation **47804c7** adds benchmark-only `ConditionedFrameCache` and
`GraphedConditionedFrameCache`. The first refresh remains Dense; later refreshes
recompute **every future-frame query**, preserve the conditioned K/V and hidden
rows, and retain all attention keys. World diagnostics are reconstructed to
their full layouts. Refresh cadence is unchanged: every step for the Dense
control, steps 0 and 5 for the temporal candidate. All ten action steps and both
original schedulers execute. There is no action-guidance, drift selection,
precision, weight, RNG, horizon, preprocessing or renderer change in this factor.

The checkpoint SHA-256 is
`6c087e5b9e201f19dbe92834b470a7ee18937a18ea9adcbb49886aa1e3ac4c61`.
Joint bf16, horizon 32, replan 10, nine video frames, seed 42 and CPU RNG are
unchanged. Python 3.10.20 / torch 2.7.1+cu126 were reused without installation or
sync. Targeted CPU tests passed **20/20 in 1.85 s**, exit 0, including changed
requests, exception cleanup, current-cadence action agreement and diagnostic
layouts. Real-checkpoint graph runs below provide CUDA execution coverage.

## Full request measurements

The first run, **47804c7**, used empty GPU 1 at **01:28:46–01:30:32 UTC**:
8 variants × 24 samples = **192 timed requests**. Commit **e453bec** then adds
only the unchanged eager/graphed guided candidate as controls, on empty GPU 2:
10 variants × 30 = **300 timed requests**. Both exit 0. Input/variant-position
combinations are fully balanced, with two warm calls per variant and three
graph-capture warmups. Timing spans synchronized `predict_action` through CPU
actions. Cold/capture and model loading are recorded separately. GPU 3 remains
unused by this work; other users' processes were not signalled.

| Complete request | First run mean ms | Guided-control run mean ms |
|---|---:|---:|
| Native Dense eager | 473.87 | 467.46 |
| Temporal eager | 288.72 | 283.36 |
| Conditioned-frame Dense eager | 473.46 | 468.63 |
| Conditioned-frame temporal eager | 288.26 | 284.05 |
| Native Dense transformer graph | 293.99 | 288.53 |
| Temporal transformer graphs | 149.77 | 146.60 |
| Conditioned-frame Dense graphs | **259.10** | **254.80** |
| Conditioned-frame temporal graphs | **145.52** | **142.87** |
| Guided 10% eager, unchanged | — | 285.44 |
| Guided 10% graphs, unchanged | — | **140.49** |

Conditioned-frame reuse gives **1.134660× / 1.132371×** additional graph-Dense
speedup and **1.029212× / 1.026136×** additional graph-temporal speedup in the two
runs. Its eager version gives no useful gain (approximately ±0.25%). Applying
it to both sides changes the temporal/Dense ratio from **1.962900× to
1.780480×** initially and **1.968094× to 1.783456×** in the guided comparison.
Guided's old-control ratio in that same run is **2.053662×**; against the
strengthened Dense control it is **1.813595×**.

All **492 timed requests**, plus **90 correctness-only requests**, passed
bitwise comparisons against their own eager reference. Conditioned-frame
variants also match their corresponding pre-factor actions exactly on every
checked input. This does **not** say temporal/guided actions equal Dense: their
existing approximations remain. The guided control's pre-factor reference is
its own unchanged eager candidate. Each conditioned Dense request reuses 26,460
token-layer rows; the interval-5 variant reuses 2,940. All 300 action-layer
updates execute. Independent raw audits reproduce means and exact sample
balance; both observed timing modes are retained without selecting a window.

In the final run loading took 15.69 s. First observed calls were 0.971 s for
native Dense graph, 0.782 s for conditioned Dense graph, 0.919 s for conditioned
temporal graph and 1.917 s for guided graph; sequential first calls are not
equal-lifecycle startup comparisons. Peak allocated/reserved memory across the
variants was 25,759 / 26,378 MiB. Raw summaries contain medians, p95, dispersion
and graph setup costs.

```bash
# Frozen worktree DreamWAM-conditioned-audit-e9cd379, diagnostic only.
PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/sparse/audit_conditioned_frame.py --out-dir "$NEW_AUDIT_DIR"

# Frozen worktree DreamWAM-conditioned-guided-e453bec, unchanged environment.
PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python scripts/sparse/benchmark_conditioned_frame.py \
  --include-guided-control --reps 30 --out-dir "$NEW_BENCHMARK_DIR"
```

The first factor run omits `--include-guided-control`, uses `--reps 24`, commit
47804c7 and GPU 1. Server artifacts are under
`dreamwam-sr/outputs/conditioned-frame-audit-20260920` and
`dreamwam-sr/outputs/conditioned-frame-factor-20260920`.
The [evidence bundle](evidence/conditioned-frame-20260920/) contains manifests,
compressed raw invariant observations with hashes, both full timing sets, raw
audits, tests, resource inventories and exit codes. Next: preserve these common
controls, complete official paired coverage, and require a separate frozen run
before adopting any new policy behavior.
