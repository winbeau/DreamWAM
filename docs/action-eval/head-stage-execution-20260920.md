# Fixed-budget Head/Stage execution comparison

Status: **MEASURED negative execution result; no SR**.
Date: **2026-09-20 UTC**.
All 300 timed requests completed at **04:31:54 UTC**, exit 0. Compact
Head × Stage takes **338.48 ms**, versus **306.95 ms** with the identical
mask: **0.9068×**, or 10.3% higher latency. Native Dense takes 303.96 ms;
the additional conditioned-frame Dense control takes **269.81 ms**. This
implementation is not promoted as a fast candidate.

Complete [comparison CSV](evidence/head-stage-execution-20260920/comparison.csv),
[raw requests](evidence/head-stage-execution-20260920/requests.jsonl),
[summary](evidence/head-stage-execution-20260920/summary.json),
[manifest and all eager actions](evidence/head-stage-execution-20260920/manifest.json),
and [completion audit](evidence/head-stage-execution-20260920/execution-audit.json)
are retained, including the initial missing-checkpoint error log.

| Budget | Masked ms | Compact ms | Compact / identical-mask speedup |
|---|---:|---:|---:|
| Full Dense | 306.85 | 309.69 | 0.991× |
| Uniform 50% future keys | 306.92 | 312.75 | 0.981× |
| Head-only, same total budget | 307.02 | 338.46 | 0.907× |
| Head × Stage, same total budget | 306.95 | 338.48 | 0.907× |

Compact 50% budgets submit 1,939,104 SDPA matrix pairs per layer, versus
2,550,624 for full-mask execution (24% less); this is a matrix-extent count,
not measured hardware FLOPs. Grouping and copying remain part of the measured
implementation, whose net latency increased. All 390 complete calls including
quality checks and warmups passed; the 300 timed outputs matched their own
eager actions bitwise. The 12 masked input/profile pairs reproduce the prior
combined experiment's actions exactly. The audit recomputed means and action
relative L2 with zero discrepancy, and verified all stage/replay counts.

The separate check input retains the prior quality pattern: normalized action
relative L2 is **0.15317 uniform, 0.08289 Head-only, 0.08386 Head × Stage**
for compact execution. Compact versus masked action L2 is nonzero, about
0.0035–0.0043; even compact full Dense has 0.00362 relative L2 versus native.
Thus compact execution is not bitwise equivalent to the mask reference.
No new SR tolerance or decision-preservation claim is inferred.

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

Source **`570f5b0`** was committed/pushed locally, pulled into the clean server
checkout and frozen at `DreamWAM-head-execution-570f5b0`. Server CPU checks
passed **12 tests**, with three CUDA checks skipped, in **12.77 s**, exit 0.
The subsequently admitted physical GPU 4 checks passed **9 tests**, including
all three released bf16/CUDA replay cases, in **18.38 s**, exit 0. The latter
set overlaps the six CPU implementation checks; these are not 21 unique tests.
Logs and GPU admission/exit records are under
`outputs/head-stage-execution-20260920/tests-{cpu,cuda}-570f5b0*` on the server.

The initial `graph-570f5b0` launch exited **1** before loading a model because
the new worktree lacked its external checkpoint link. Its log, admission and
exit file are retained. Adding a symlink to the unchanged authoritative
checkpoint fixed this data-mount omission; application code and weights did
not change. Subsequent resource admission polls preserved one available card.

The actual benchmark manifest began **2026-09-20 04:28:30 UTC**, collector
PID **403422**, on GPU 4 (`GPU-490b4a76-6210-31b9-4e03-838a113cf5f4`),
with GPU 2 left unused by this task at admission. At **04:28:53 UTC** the
collector was confirmed live, still in model preparation. Runtime artifacts:
`/data/chenjiayu/wenbiao_zhao/dreamwam-sr/outputs/head-stage-execution-20260920/graph-570f5b0-rerun/`.
The command above used those exact completed-calibration paths and this output
directory. The outer `.log`, `.pid`, `.exit` and `-admission.json` preserve the
process and shared-host resource history. The completed results are above.

Incomplete runs and negative measurements are retained.
This remains an exposed-input proxy study; complete official paired SR and
M2 bridge comparisons remain outstanding.
