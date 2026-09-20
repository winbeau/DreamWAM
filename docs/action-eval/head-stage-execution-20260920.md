# Fixed-budget Head/Stage execution comparison

Status: **PLANNED; no new speed or SR result**. Date: **2026-09-20 UTC**.
This follows the completed [offline classification and allocation test](head-stage-calibration-20260920.md).
The single factor is execution of those exact masks: expanded-mask joint SDPA
versus per-budget head groups with physically shorter future-video K/V inputs.
Uniform, Head and Head × Stage all keep the same 168 future blocks per
layer/stage. No AV selection, temporal visual reuse, FFN pruning or policy
adapter change is introduced.

The compact path computes conditioned and action rows against all joint keys,
with their original visibility. Future-video queries use conditioned keys plus
the same uniformly spaced future blocks as the completed diagnostic. Fixed
head groups gather only the relevant tensors. Full-budget compact execution
also uses the split implementation; its output drift and overhead are retained.
Shorter matrix extents do not imply net speedup, since grouping, gathers,
scatters and extra launches all cost time.

`benchmark_head_stage_execution.py` uses the same three hash-verified real
observations, the same two-input classification, checkpoint and released
settings. It measures ten variants in balanced rotating order: native Dense,
masked/compact Dense, masked/compact uniform, masked/compact Head,
masked/compact Head × Stage, and the existing conditioned-frame Dense control.
Graph runs use the same three-stage transformer replay scope for the nine
direct comparisons; conditioned Dense retains its already verified graph
implementation as an additional stronger reference. The stage bounds remain
`[0,3)`, `[3,7)`, `[7,10)`. Eager and graph timing are separate runs.

Each variant first gets its own eager output on all three inputs. Graph/buffer
replay must reproduce those actions, normalized actions and final video latents
bitwise, including input 0 after visiting the other inputs. Timed requests
retain whole `predict_action` synchronization and CPU action output, include
selection/copy/replay costs, and omit diagnostic output observers. Thirty
repetitions per variant balance three inputs and ten order positions. Cold
setup is reported separately. Dense expanded-mask parity is mandatory; compact
versus masked rounding differences are explicitly measured, not assumed zero.

Tests compare compact outputs to the independent existing mask implementation,
inspect the actual SDPA query/key/head extents, preserve all action keys and
exercise stage and request boundaries. Real bf16 geometry uses predeclared
attention-output tolerances: relative L2 at most 0.006 and maximum absolute
error at most 0.016, plus bitwise own-eager CUDA replay. These are kernel
correctness checks, not an SR tolerance or a policy-equivalence claim.

Run only in an admitted server GPU window, preserving the last available card
and avoiding dedicated rendering cards. Reuse Python 3.10.20 / torch 2.7.1+cu126;
no dependency install or sync. Checkpoint SHA256:
`6c087e5b9e201f19dbe92834b470a7ee18937a18ea9adcbb49886aa1e3ac4c61`.

```bash
# Pinned worktree; paths from the completed calibration manifest.
PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES="$ADMITTED_GPU" OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  scripts/sparse/benchmark_head_stage_execution.py --mode graph \
  --inputs "$CALIBRATION_ROOT/inputs-gpu4/manifest.json" \
  --classification "$CALIBRATION_ROOT/analysis-f489335/classification.json" \
  --out-dir "$EXECUTION_OUTPUT"
```

Source commit, admission, commands, exit codes and measured artifacts will be
recorded after execution. Incomplete runs and negative measurements are retained.
This remains an exposed-input proxy study; complete official paired SR and
M2 bridge comparisons remain outstanding.
