# Guided graph full Spatial: independent coverage in progress

Status: **STOPPED at native-renderer stagnation limit; 0/500 terminal outcomes**.
A new full run of the same policy as the
completed [15-episode guided graph pilot](visual-graph-dispatch-single-factor-20260919.md)
started **2026-09-20 00:54:36 UTC**. It is a separate 500-identity matrix, not an
extension or aggregation of earlier pilot or eager outcomes. The two accepted
failures in the eager guided run remain intact. The full matrix is needed to
measure the quality tradeoff rather than infer overall SR from those two cases.

## Frozen deployment

- Model: **2c02c5c**, `DreamWAM-graph-policy-2c02c5c`.
- Config: action-eval **35d0366**,
  `configs/experiments/dreamwam-guided-visual-graph-libero-spatial.yaml`, frozen in
  `action-eval-guided-graph-full-35d0366`.
- Evaluator: unchanged **4701ac2**, `action-eval-visual-4701ac2`.
- Checkpoint SHA-256:
  `6c087e5b9e201f19dbe92834b470a7ee18937a18ea9adcbb49886aa1e3ac4c61`.
- Config fingerprint:
  `4a06c9d185d1496f0c8e547c5db3f99ce84795c486bee4ce2c88b6ad104c5c06`.
- Run: `guided-visual-graph-libero-spatial-20260920/gpu4/run-20260920T005437Z-4a06c9d1`.

Options are interval 5, token keep ratio 0.1, action guidance weight 1.0 and
`graph_dispatch: all_transformers`. Neither experimental K/V staging nor DiT
boundary capture is enabled. Joint bf16, ten denoising steps, horizon 32,
max_steps 400, wait 30, replan 10, seed 42, 256×256 simulator observations and all
10 tasks × 50 initial states are unchanged. The startup policy fingerprint
reports the expected visual options and checkpoint. All full-run manifests were
independently checked for identical benchmark, protocol and planned identities.

The existing environment was reused without installs or synchronization.
`validate` and `doctor` both exited 0 before launch; doctor is not a model smoke.
The matching guided graph pilot had already completed all 15 identities.
GPU 4 (`GPU-490b4a76-6210-31b9-4e03-838a113cf5f4`) was explicitly shared: launch
inventory showed 11% utilization, 43,145 MiB used and 100,012 MiB free. This work
does not allocate GPU 3 and does not signal independently owned processes.

```bash
# From the frozen evaluator; use the frozen config and model paths above.
export MODEL_ROOT LIBERO_ROOT GPU_UUID
export CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=1 MUJOCO_GL=egl PYTHONFAULTHANDLER=1
unset MUJOCO_EGL_DEVICE_ID
export PYTHONPATH="$PWD/src:$LIBERO_ROOT"
.venv/bin/python -m action_eval.cli validate "$CONFIG"
.venv/bin/python -m action_eval.cli doctor "$CONFIG"
.venv/bin/python -m action_eval.cli run "$CONFIG" --output "$RUN_PARENT/gpu4"
```

## Initial native abort and bounded recovery

The initial process exited **134** in robosuite `read_pixels` before any episode
reached an official terminal outcome. It is a renderer error, not a task failure.
The previous policy process was verified exited before recovery. Helper
**f8eae67**, SHA-256
`eb26dd9f276c5fa87bed6658b64c81cb040320f551eb99896e08b7af67e177c2`,
started `gpu4-recovery-01` with at most 20 invocations and **two consecutive
no-progress recovery invocations**. With the initial zero-progress abort, this
stops after three such invocations if neither recovery makes progress. The same
config, manifest and every subsequently accepted result remain immutable;
non-native errors and resource-gate closures stop the controller. A stagnation
cap must not be reset simply to keep retrying unchanged conditions.

At **2026-09-20 01:00:04 UTC**, the guided graph run still had **0/500 terminal
outcomes** and its recovery supervisor **225929** was live. Its startup options
are verified, but no SR or full-episode quality claim is made. Original logs,
preflights, manifests, startup provenance and counts are retained in the
[evidence bundle](evidence/guided-graph-full-20260920/), with authoritative server
artifacts under `dreamwam-sr/outputs/guided-visual-graph-libero-spatial-20260920`.

## Stagnation cap reached

Both recovery invocations also exited with native SIGABRT and no new terminal
outcomes. Together with the initial run, three consecutive invocations made zero
progress. The helper stopped with `stagnation_limit`, exit 1, and supervisor
225929 was confirmed absent. This is **0 successes, 0 task failures, 0/500
coverage**, not a measured task failure rate. The run is not restarted under
unchanged conditions. The [stagnation evidence](evidence/guided-graph-full-20260920/stagnation/)
retains both error logs, per-attempt counts, summary and a fresh progress audit.

At **01:04:51 UTC**, the productive full runs had Dense **372/500** terminal
outcomes (365 successes, seven task failures), eager temporal **199/500 successes**
and temporal graph **158/500 successes**. Their live supervisors were respectively
**229197** (Dense batch 07), **217246** and **223555**. Dense batch 06 had stopped
at a transient resource gate with 369 outcomes; GPU 0 was later verified empty
at 0% utilization before batch 07 resumed. Accepted outcomes remain unchanged.
These three runs continue; the overall SR-constrained acceleration goal is still
active and incomplete.

## Earlier full-run snapshot

The same timestamped audit found:

| Frozen full run | Terminal / planned | Successes | Task failures | Recovery supervisor |
|---|---:|---:|---:|---:|
| Matched Dense, GPU 0 | 368 / 500 | 361 | 7 | 213211 |
| Eager temporal, GPU 1 | 184 / 500 | 184 | 0 | 217246 |
| Temporal graph, GPU 2 | 144 / 500 | 144 | 0 | 223555 |

All three controllers were live. Temporal graph batch 02 stopped after seven
productive invocations with 132 successes because the resource check observed
35% utilization despite all model memory having been released. A later check
found GPU 2 empty at 0%, and batch 03 resumed only after the old supervisor exited.
The source/config hashes and accepted outcomes did not change.

Eager guided still retains 95/500 outcomes (93 successes, two official task
failures), inactive. Its failures are `t000-i048-r00-s42` and
`t001-i036-r00-s42`; both Dense and the temporal variants succeeded on those
identities. Initial camera hashes are not repeatable, so these paired outcomes
do not isolate the cause. No partial run supplies a complete SR or fills another
run's denominator. Next: continue productive bounded recoveries, respect any
stagnation caps, and compare official outcomes only after complete coverage.
