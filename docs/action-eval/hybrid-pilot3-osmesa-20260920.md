# H100 / OSMesa hybrid pilots: speed and benchmark-owned outcomes

Status: **Both pilots COMPLETE; KV25 negative, KV75 promising but unconfirmed**.
Date: 2026-09-20 UTC. User request: run a few paired episodes first, targeting
>1.5× versus equally optimized Dense with an acceptable SR; no numeric SR
tolerance is assumed. These are development pilots, not benchmark reproduction
or independent confirmation, and not the deferred 50-pair experiment.

## Frozen scope and admission

Each pair of arms runs Spatial tasks **0, 1, 2 × initial state 0**, once per
identity. The second pair repeats these already-exposed identities; repeated
Dense outcomes cannot be pooled as six independent episodes. The KV75 follow-up
was proposed after the first KV25 failure, so it is explicitly adaptive screening.

Unchanged `dreamwam-release-v1`: max_steps 400, wait_steps 30, resolution 256²,
replan_steps 10, seed 42, fixed-per-predict RNG, 32-action horizon, 10 denoising
steps, original checkpoint/precision. Dense uses exact prompt reuse and exact
conditioned-frame reuse, with all-transformer graphs. Both hybrids use the same
prompt optimization, CUDA graphs, D at step 0, S at step 5, R elsewhere,
10% visual recompute, action-drift weight 1 and balanced frame quotas.
The only candidate option changed in the follow-up is the K/V read budget:
25% → 75%. Step 5 is a fixed diagnostic control, not a searched best location.

Policy uses physical H100 GPU 3,
`GPU-c0af33a9-498c-ff7c-bb56-e9992ccded30`. Rendering uses private CPU OSMesa,
`LP_NUM_THREADS=4`; other CPU thread limits remain 1. GPU 4/5 are not used by
this task. Occupancy is checked before each arm and journalled. This is a shared
host, not an exclusive hardware reservation. No dependencies or simulator
versions were changed. EGL is not used.

The frozen policy worktree is `DreamWAM-hybrid-pilot-5d86765`, source
`5d867657a0a4dd1bc2b6a45d17c386c71867c301`. It reuses the verified interpreter
from `DreamWAM-fresh-6c52f36/.venv/bin/python`, without loading that old model
checkout's source. Python 3.10.20 / torch 2.7.1+cu126 / CUDA 12.6; robosuite 1.4.1,
MuJoCo 3.3.2. Checkpoint SHA256:
`6c087e5b9e201f19dbe92834b470a7ee18937a18ea9adcbb49886aa1e3ac4c61`.

Evaluator sources/configs: `d67cab193f949c9a8a0387d42216396c5a6962ec`
for KV25, and `b751d9ad8b2ccdaf9b4711fc7caca0f99f23cd1a` for KV75
(the latter only adds the KV75 configuration).
Both use separate frozen worktrees and the existing evaluator environment.
Config entrypoints in action-eval:

- `configs/experiments/dreamwam-hybrid-dense-pilot3-osmesa.yaml`
- `configs/experiments/dreamwam-hybrid-compact25-pilot3-osmesa.yaml`
- `configs/experiments/dreamwam-hybrid-compact75-pilot3-osmesa.yaml`

`MODEL_ROOT`, `MODEL_PYTHON`, `LIBERO_ROOT` and `GPU_UUID` are explicit inputs.
The fresh-token OSMesa environment script is sourced for graphics/thread setup;
its old `MODEL_ROOT` is explicitly replaced with the frozen hybrid checkout.

## Results

Warm means omit the first request of **each** episode and require a recorded
prompt-cache hit. This rule was defined before reporting the outcomes. Times
cover complete model `predict_action` through CPU action output, excluding IPC,
model load and CPU simulation. IPC-inclusive and first-request timings are
reported separately. Closed-loop trajectories and request counts diverge, so
these ratios are descriptive, not same-observation causal timing comparisons.

| Pair | Arm | Successes | Warm requests | Mean warm model time | Dense / candidate |
|---|---|---:|---:|---:|---:|
| KV25 | Strengthened Dense | 3/3 | 26 | 262.89 ms | — |
| KV25 | Hybrid compact 25% | 0/3 | 117 | 140.94 ms | 1.865× |
| KV75 | Strengthened Dense | 3/3 | 26 | 271.72 ms | — |
| KV75 | Hybrid compact 75% | 3/3 | 26 | 149.24 ms | 1.821× |

KV25's IPC-inclusive warm mean is 149.06 ms versus Dense 271.53 ms, **1.822×**.
Its p50/p95 model times are 140.88/144.55 ms versus Dense 262.64/265.28 ms.
Episode-balanced warm speedup is 1.865× as well. Do not use the 2.013× ratio
from averages including cold calls: the failed candidate executes many more
requests and therefore amortizes its initial setup over a different denominator.

KV25 Dense succeeded at steps **73, 107, 92**. Hybrid exhausted **400 steps on
each task**, with 40 policy calls per episode. There were **no model/environment
errors**, no retries, no forced cleanup and no renderer read failures. Mean whole
episode time is 28.21 s Dense versus 84.07 s Hybrid: faster requests did not make
task completion faster. The controller exited 0 with `PILOT_PAIR_COMPLETE`.

On these three pairs, success difference is −100 percentage points; Wilson95
intervals are [43.85%, 100%] for 3/3 and [0%, 56.15%] for 0/3. These tiny-cohort
intervals do not certify generalization. Exact McNemar one-sided p=0.125;
no significance or non-inferiority claim is made. A task-stratified bootstrap
cannot estimate within-task variation from one episode per task, so the audited
report withholds that interval. The original report containing degenerate
bootstrap intervals is retained, superseded by `paired-sr25-audited.json`.

KV25 first policy inputs matched byte-for-byte on all three pairs: **3/3 state,
3/3 agentview, 3/3 wrist hashes equal**, after the prescribed wait and before any
policy-dependent action. Native RGB guard logs show **754/754 Dense reads** and
**2610/2610 Hybrid reads** completed, with no unwritten-buffer detections.
All 120 Hybrid calls execute the requested 1 Dense / 1 Sparse / 8 Reuse steps
and all 300 action-layer updates. The request cache/graph checks are necessary
implementation evidence, not a substitute for the observed failures.

KV75 succeeds at steps **75, 109, 96**, versus **73, 107, 92** for its new Dense
control. All six outcomes are terminal successes, with zero errors/retries and
normal cleanup. Its 29 requests retain 26 warm samples, exactly matching the
Dense call counts. IPC-inclusive warm means are **158.46 vs 281.72 ms (1.778×)**.
Model p50/p95 are **148.58/155.46 ms** versus Dense **265.62/268.44 ms**; the Dense
mean retains a latency outlier instead of silently dropping it. Episode-balanced
warm model speedup is 1.830×.

**Cold/setup and whole-episode limits matter:** including the first request of
each episode gives model means **231.29 vs 328.06 ms, only 1.418×**, on this short
29-request sample. Mean first requests are 942.41 ms Hybrid vs 816.34 ms Dense;
they include prompt misses and any graph construction. Mean episode walls are
**27.99 s Hybrid vs 27.32 s Dense**. Thus this pilot supports >1.5× **warm
inference**, not >1.5× cold-inclusive or rendered end-to-end operation. The
extra execution steps and CPU simulator/rendering costs remain visible.

KV75's pilot success difference is 0 pp. Each 3/3 Wilson95 interval is still
[43.85%, 100%]; three pairs cannot establish SR preservation or non-inferiority.
Native reads completed **754/754 Dense** and **770/770 Hybrid**. First policy
inputs again match all 3/3 state and both-camera hashes. Repeated Dense actions
across the two cohorts are byte-identical on all three episodes, with the same
termination steps; these are repeatability checks, not additional independent
SR samples. All 29 KV75 calls perform 300 action-layer updates and 9,720 visual
token-layer recomputations; post-anchor attention physically reads **221/294**
visual K/V rows (KV25 reads 74/294).

KV25 finished at **13:15:50 UTC**; KV75 at **13:22:59 UTC**, both controllers exit 0
with `PILOT_PAIR_COMPLETE`. No active evaluation was left behind. The wider
read support is the next candidate to retain; do not freeze it as the final
method or infer that step 5 is optimal from this pilot.

### Short same-input performance recheck for KV75

After both closed-loop pairs finished, a separate, finite **36-request** replay
used the three historical hash-verified debug observations, six repetitions and
two warmups, interleaving the same strengthened Dense and KV75 on GPU 3.
Source remains frozen `5d86765`; completed **36/36**, exit 0, at **13:26:16 UTC**.
All 18 timing pairs are from the **same invocation**, without resume. Loading,
graph setup and prompt misses are reported separately, not hidden in warm means.

Dense mean **261.98 ms**, KV75 **147.63 ms**, paired **1.775×**. Candidate p50/p95
are 142.71/158.13 ms; its 227.76 ms maximum is retained. All 36 saved actions were
reloaded and verified byte-identical to their own eager references, with all 300
action-layer updates. This supports the warm speed direction on identical inputs
but is still a small shared-host debug experiment, not a held-out quality result.
Mean action relative L2 is 0.0814; SR comes only from the separate closed loop.
Manifest identity:
`c58c5a45a98115c9ae4685a48bb758d252aff0c35e157a77747bb21bd2836c84`.

## Implementation and validation for the pilot

All changes followed local edit → commit/push → clean server pull. Active policy
and evaluator checkouts remained frozen while reporting tools were developed.

- action-eval `6b8a018`: explicit three-episode configs; opt-in action artifacts
  now retain per-call model timing, IPC wall time and executed diagnostics.
  Success, actions and protocol logic are unchanged; deterministic action/input
  artifacts stay separate from nondeterministic timings.
- DreamWAM `5d86765`: bounded controller accepts explicit config pairs/counts,
  verifies matching protocols/resources/common options and effective fingerprints;
  real-checkpoint adapter verification accepts Hybrid configs.
- `234d22e`: separate warm/cold/IPC timing audit with complete-coverage checks.
- `446741c`: withhold uninformative one-init task bootstrap intervals and flag
  degenerate intervals for larger fully observed samples.

Verification (server only, exit 0): action-eval **200 tests**; DreamWAM
**312 passed, 15 CUDA-skipped, 8 subtests** at reporting source `446741c`.
Both Hybrid budgets separately passed 14 adapter requests (7 per arm) against
their own eager outputs, including changed observations/instructions, prompt
hits/misses, reset and cleanup. `validate`, `doctor`, schema equality and the
real OSMesa `describe` action/observation check passed before rollout.

An initial command used `python -m action_eval`, but this package has no
`__main__`; it exited before model/environment execution. The installed
`action-eval` entrypoint was then used successfully. This was a command error,
not an episode, a retry or a model failure.

## Reproduction and evidence

H100 output root:
`/root/wenbiao_zhao/dreamwam-sr/outputs/hybrid-pilot3-20260920/`.
`pair/` is the KV25 matched cohort; `pair75/` is its independently run follow-up.
Each contains a bounded `controller.json`, resource journal, exact arm commands,
timestamps/exits, native-read logs, frozen manifests, provenance, accepted
result JSON, full predictions/executed actions, initial-input hashes and per-call
timing/diagnostic JSON. `adapter{,75}/report.json` records the real-checkpoint
preflight, including candidate configuration hashes and own-eager parity.
Checked-in result/provenance hashes:
[hybrid-pilot3-evidence-20260920.json](hybrid-pilot3-evidence-20260920.json).

Controller invocation (server, after fresh resource admission):

```bash
"$EVAL_PYTHON" "$MODEL_ROOT/scripts/sparse/run_fresh_token_pair.py" \
  --eval-root "$EVAL_ROOT" --model-root "$MODEL_ROOT" \
  --adapter-report "$PILOT_OUT/adapter/report.json" --out-dir "$PILOT_OUT/pair" \
  --dense-config "$EVAL_ROOT/configs/experiments/dreamwam-hybrid-dense-pilot3-osmesa.yaml" \
  --sparse-config "$EVAL_ROOT/configs/experiments/dreamwam-hybrid-compact25-pilot3-osmesa.yaml" \
  --planned-episodes 3 --render-backend osmesa --first-arm dense \
  --authorized-gpus 3 4 5 --wall-seconds 900 --admission-seconds 120 --stop-grace-seconds 180
```

For KV75 use its frozen evaluator root/config, adapter75 report and `pair75` output.
No command shrinks a 50-episode manifest or reuses historical accepted outcomes.
Model options are mutually exclusive; the fresh-token method is not stacked onto
Hybrid. The repository config/adapter skills guided this separation of model,
configuration and benchmark-owned success.

The additional same-input run used `generate_hybrid_schedules.py` with
`--candidate-sparse-steps 5 --refresh-counts 1 --recompute-ratio 0.10
--read-mode compact --read-ratio 0.75 --frame-quota balanced --backend cuda_graph`,
then `benchmark_hybrid_schedules.py --split-label debug --reps 6 --warmup 2
--group-size 1 --controls dense_strong --allow-shared-gpu`. The complete resolved
command, input hashes, hardware inventory and setup costs are in
`same-input75/manifest.json`, with stdout in `same-input75.log`.

Postprocessing:

```bash
python3 scripts/sparse/paired_sr.py --dense "$PILOT_OUT/pair/dense/run" \
  --sparse "$PILOT_OUT/pair/sparse/run" --label hybrid-compact25-pilot3 \
  --json "$PILOT_OUT/paired-sr25-audited.json"
python3 scripts/sparse/summarize_policy_timings.py --dense "$PILOT_OUT/pair/dense/run" \
  --sparse "$PILOT_OUT/pair/sparse/run" --out "$PILOT_OUT/timings25.json"
python3 scripts/sparse/audit_initial_inputs.py --reference "$PILOT_OUT/pair/dense/run" \
  --candidate "$PILOT_OUT/pair/sparse/run" --out "$PILOT_OUT/inputs25-audit.json"
```

The complete 46/512 schedule search, development-cohort expansion and independent
confirmation remain unexecuted. This pilot does not identify a best refresh step.
