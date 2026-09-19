# Joint temporal visual reuse: speed passed, SR not established

Status: **MEASURED latency; both paired pilots complete; full Spatial SR incomplete**.
Date: 2026-09-19 UTC. Status snapshots below include their own timestamps.
The complete goal (SR-constrained ≥1.5×, aiming for 2×) remains active and unproven.
Subsequent [token refresh and action-guidance factors](visual-token-cache-single-factor-20260919.md)
are measured separately and do not change the frozen V1 full Spatial candidate.

The preceding [FFN-only factors](ffn-cache-single-factor-20260919.md) did not provide
useful speedup. This next experiment changes one broader factor: **how often all
visual transformer layers refresh within a request**. It does not combine token FFN
recomputation, neuron selection, sparse VV attention, or public operator changes.

## Execution and controls

[`VisualStepCache`](../../dreamwam/sparse/visual_step_cache.py) refreshes video
layers at steps 0 and 5. A refresh runs the original Joint computation, recording
each layer's post-RoPE visual K/V and final visual hidden states, including world
residuals. Steps 1–4 and 6–9 reuse those visual results. Every action layer still
executes, using the model's existing action-with-video-cache path with dense AV+AA
and the original joint softmax normalization. The original video pre/post-DiT,
both schedulers, separate original RNG generators, and all ten action updates run.
Cached world residual effects come from refresh steps; this is an approximation
to the current visual branch, and requires its own official quality validation.

This is a coarse temporal visual-token reuse ablation. **It is not yet action-guided
selection, and it is not rMuscle's full cache system.** Positive speed is a reason
to test its quality, not evidence that quality is preserved.

Controls: native Dense and the identical wrapper at `refresh_every=1`, which must
be bitwise native Dense. Exact checkpoint, bf16, 9 video frames, horizon 32,
replan 10, 10 denoising steps, seed 42, CPU RNG, image size and gripper semantics
are unchanged. Checkpoint SHA256:
`6c087e5b9e201f19dbe92834b470a7ee18937a18ea9adcbb49886aa1e3ac4c61`.

## Complete policy-request measurement

Commit `aaf907a`, separate fixed Git worktree, H200 NVL GPU 6
(`GPU-8cf0627c-5919-bace-1c42-e5d627c36cb5`), ending **20:53:36 UTC**, exit 0.
Existing Python 3.10.20 / torch 2.7.1+cu126 environment; no dependency installation
or changes. Two warmups and 24 requests per variant, rotating execution order,
same three synthetic input identities as the FFN screening. Each timer covers
the entire `predict_action`, from text/VAE encoding to CPU action output.

| Variant | Mean ms | p50 ms | p95 ms | Speedup vs matched mean |
|---|---:|---:|---:|---:|
| Native Dense | 449.87 | 449.46 | 455.56 | — |
| Matched Dense, refresh every step | 455.16 | 454.91 | 460.55 | 1.000× |
| Visual refresh at steps 0 and 5 | 275.10 | 275.26 | 278.03 | **1.655×** |

Speedup against native Dense is **1.635×**. Full-budget actions are bitwise equal
on all three inputs and on all measured Dense requests. Model parameter versions
are unchanged. Each candidate request performs **60 visual layer updates and
300 action layer updates**, compared with 300 and 300 for Dense. Seven original
execution tests passed, including direct counting of both schedulers and both
experts; later policy/adapter/coverage checks passed 21 tests.

Maximum synthetic-input action relative L2 is 0.10035. This is not SR evidence.
The measurement is a complete **policy** request, not evaluator transport latency
or a simulator rollout measurement.

Evidence: [manifest](evidence/visual-cache-20260919/refresh5-manifest.json),
[72 individual requests](evidence/visual-cache-20260919/refresh5-requests.jsonl),
[summary](evidence/visual-cache-20260919/refresh5-summary.json),
[GPU telemetry](evidence/visual-cache-20260919/refresh5-gpu.csv).
Server path: `dreamwam-sr/DreamWAM-visual-aaf907a/outputs/visual-20260919/refresh5`.

```bash
PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=6 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python scripts/sparse/benchmark_ffn_context_cache.py \
  --cache-kind visual_steps --refresh-every 5 --warmup 2 --reps 24 \
  --out-dir outputs/visual-20260919/refresh5
```

## Factor V2: one visual refresh, ten action steps

A separate run changes only `refresh_every` from 5 to 10: step 0 computes visual
layers, and steps 1–9 reuse them. It still executes all ten action updates and
both schedulers. No FFN selector, action guidance, sparse attention or public
operator optimization is combined with this factor. This run uses the original
fixed **aaf907a** implementation, checkpoint and synthetic inputs.

It completed **21:36:10 UTC**, exit 0, on H200 NVL GPU 4
(`GPU-490b4a76-6210-31b9-4e03-838a113cf5f4`), with two warmups and 24 measured
requests per variant. GPU 4 had 0% utilization and 48,981 MiB occupied
at launch; a foreign multi-GPU process remained present. This is explicitly a
shared-card measurement. Native/matched/candidate request standard deviations
were 2.63/3.04/2.30 ms; all samples and telemetry are retained.

| Variant | Mean ms | p50 ms | p95 ms |
|---|---:|---:|---:|
| Native Dense | 459.96 | 459.51 | 465.07 |
| Matched Dense | 464.47 | 463.39 | 469.64 |
| Visual refresh at step 0 only | 257.55 | 256.81 | 261.06 |

The measured speedup is **1.803×** matched Dense (**1.786×** native Dense).
Each request performs 30 visual and 300 action layer updates. Full-budget bitwise
parity and unchanged parameter versions pass. Maximum synthetic action relative
L2 is 0.13164. **V2 has no closed-loop quality evidence** and does not replace
the frozen interval-5 candidate being evaluated for SR.

An earlier attempted invocation on GPU 1 was cancelled (owned benchmark PID
93656, exit 143) because its launch snapshot reported 100% utilization before
our model load. It supplies no latency result and was not counted as a strategy
failure. No foreign process was signalled.

Evidence: [manifest](evidence/visual-cache-20260919/refresh10/manifest.json),
[72 requests](evidence/visual-cache-20260919/refresh10/requests.jsonl),
[summary](evidence/visual-cache-20260919/refresh10/summary.json),
[GPU telemetry](evidence/visual-cache-20260919/refresh10/gpu.csv),
[cancelled invocation](evidence/visual-cache-20260919/refresh10/cancelled-gpu1.json).

```bash
PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=4 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python scripts/sparse/benchmark_ffn_context_cache.py \
  --cache-kind visual_steps --refresh-every 10 --warmup 2 --reps 24 \
  --out-dir outputs/visual-20260919/refresh10-gpu4
```

## Official evaluator plumbing pilot

The model option is `policy.options.visual_cache: {refresh_every: 5}`; matched
Dense uses `{refresh_every: 1}`. Omission preserves the native default. The adapter
publishes the executed option in its fingerprint and records dense/reused visual
step and visual/action layer counts per prediction. Every request, including an
exceptional request, clears cached tensors. Policy close removes wrappers before
releasing the model.

Frozen model commit **28845c6**, action-eval commit **526ad25**. Configuration
validation and doctor passed (doctor does not render). Each variant plans the same
15 episodes: LIBERO-Spatial tasks 0/1/2 × initial states 0/1/2/3/4, repeat 0,
seed 42. Both use `dreamwam-release-v1` (400 steps, wait 30, replan 10, resolution
256², official `first_success_after_wait`). No failures are retried. Native
errors remain errors; any recovery must target those same incomplete identities.

Candidate runs on physical GPU 6; matched Dense runs on physical GPU 0
(`GPU-0b850a78-f1db-6bda-8dd5-d96296bc1bfc`). A foreign job arrived on both cards
at launch. Utilization was low and memory sufficient, consistent with the user's
authorization to share lightly occupied cards. The pilot is marked shared;
the speed claim above uses the earlier isolated measurement. No foreign process
was signalled.

Server output root: `dreamwam-sr/outputs/visual-cache-pilot-20260919`, with
`refresh5/` and `matched/` run directories, matching `.log` and eventual `.exit`
files. Initial wrapper PIDs: 66056 (candidate), 66057 (matched); these are process
handles to recheck, not proof a job is still running.

The pilot is a development/plumbing check, **not full benchmark coverage**. No
official benchmark SR or non-inferiority claim is made. After pilot inspection,
freeze the candidate and evaluate complete official episode pairing; evaluate any
action-guided refresh selection as an additional isolated factor.

## Recovery and complete-suite continuation

The initial candidate process completed five successful episodes and then exited
134 in the native renderer's `read_pixels`; initial matched Dense exited 134
before a terminal outcome. These are environment interruptions, not policy
failures. All original outputs and logs remain. `action-eval resume` targets the
same frozen manifest and skips every accepted success or failure. No completed
episode was rerun during recovery, and protocol/model/GPU settings did not change.

Initial provenance was written before policy load and only finalized after the
episode loop. A native abort therefore lost the executed policy description.
Action-eval **d91ccc9** persists the description immediately after worker startup,
before renderer creation. Its full platform tests passed and the generated schema
was unchanged. The first-attempt provenance files are retained separately; their
missing description is not presented as an executed fingerprint. Recovery uses
this evaluator revision with the original frozen model **28845c6**.

At **21:28:25 UTC**, the candidate had **15 accepted successes, 0 task failures**
and exited 0. It required three recoveries after three native aborts. Matched Dense
had **9 accepted successes, 0 task failures**, four native aborts and a fourth
recovery in progress. These are pilot counts, not a complete paired SR. The
executed descriptions on both recovered lanes verify the same checkpoint hash,
10 denoising steps, horizon 32, replan 10 and fixed model seed 42; visual intervals
are 5 and 1 respectively. Consecutive recoveries without new terminal outcomes
are limited; rendering errors cannot be silently converted to failures or removed
from the planned denominator.

Evidence: [timestamped paired progress](evidence/visual-cache-pilot-20260919/progress-212825Z.json),
[candidate planned manifest](evidence/visual-cache-pilot-20260919/refresh5/manifest.json),
[executed provenance](evidence/visual-cache-pilot-20260919/refresh5/provenance.json),
[candidate episode records](evidence/visual-cache-pilot-20260919/refresh5/per_episode.csv),
[candidate pilot summary](evidence/visual-cache-pilot-20260919/refresh5/summary.json).
The server also retains all executed actions, complete predictions and policy-input
hashes per accepted episode. No rendering workaround or dependency change was used.

Complete Spatial configurations are committed in action-eval **4701ac2**:
`dreamwam-visual-refresh5-libero-spatial.yaml` and
`dreamwam-visual-matched-libero-spatial.yaml`. Each fixes all 10 tasks × 50 initial
states, repeat 0 and seed 42, with the same published 400-step protocol. Both
configuration validation and doctor passed; their runtime source matches d91ccc9.
The preflight records are under server
`dreamwam-sr/outputs/visual-cache-spatial-20260919/preflight/`.

The candidate full Spatial run started **21:27:57 UTC** on GPU 6, using the same
model **28845c6** and interval 5:
`refresh5/run-20260919T212757Z-30c49703` under that output root. Its actual worker
description was verified before rollout at **21:29:09 UTC**. The initial wrapper
PID was 90534; inspect the current process and `.exit` file before calling it live.
This is a separate 500-episode experiment; the 15 pilot records are not imported
into it. Matched-Dense full Spatial remains prepared pending pilot recovery.
**Full Spatial SR and any non-inferiority conclusion remain null/unestablished.**

Historical state, **21:38:03 UTC**: matched pilot remained at **9/15** after
three consecutive recoveries added no terminal outcomes; full candidate Spatial
remained at **0/500** after its initial process and two recoveries all exited 134
without an outcome. Further blind retries were stopped at that finite limit.
All six matched-pilot invocations and all three full-candidate invocations ended
in native aborts. Their runner/worker PIDs were confirmed absent. The completed
interval-5 candidate pilot remains 15/15 accepted successes; no formal paired SR
has been produced. Full matched-Dense Spatial was prepared but never launched.

Evidence: [recovery limit and process exits](evidence/visual-cache-spatial-20260919/recovery-limit-213803Z.json),
[full planned manifest](evidence/visual-cache-spatial-20260919/manifest.json),
[executed full-run provenance](evidence/visual-cache-spatial-20260919/provenance.json).
The active acceleration goal is not complete. The next quality gate is reliable
official rollout execution under unchanged scientific settings. Read-only inventory
of the alternate L40 server found available GPUs, but the limited search did not
find an existing DreamWAM deployment; no environment migration was performed.

### Continued progress after the resource window changed

At **21:52:35 UTC**, the foreign training workload left GPUs 0–3. The matched
pilot was resumed after this external-state change: resume 6 added four terminal
successes before a native abort, and resume 7 added the remaining two and exited
0. Both pilot sides now have **15 accepted successes, 0 task failures**, with all
15 planned identities paired and no discordant outcomes. This is a plumbing
pilot, not full-benchmark SR or evidence of non-inferiority. No settled outcome
was rerun. [Complete paired report](evidence/visual-cache-pilot-20260919/paired-pilot-complete.json)
and [Dense records](evidence/visual-cache-pilot-20260919/matched/per_episode.csv)
retain the full pilot denominator; bootstrap intervals on this small all-success
sample must not be interpreted as certainty.

A new full candidate run started on GPU 1 at **21:53:51 UTC**:
`refresh5-gpu1/run-20260919T215351Z-7f224961`. The earlier GPU-6 run remains
separate at 0/500; its attempts are not pooled into the new manifest. Full matched
Dense started on GPU 0 at **22:08:34 UTC**:
`matched/run-20260919T220834Z-87c8f1d0`. Both keep model **28845c6** and evaluator
**4701ac2**, the same checkpoint and all 500 official identities. Native renderer
aborts still occur, but each lane is making terminal progress.

Action-eval **f8eae67** adds a bounded recovery supervisor around the unchanged
4701ac2 `resume` command. It hashes frozen config/manifest and settled outcome
records, protects accepted **successes and failures**, and only continues after
SIGABRT. It stops after 20 invocations, three consecutive invocations without new
terminal outcomes, another error, or loss of the GPU resource window. Every
attempt records runner PID, exit code, outcome counts and the preceding executed
provenance. An OS lock prevents duplicate supervisors, and prior workers must exit
before a new resume. The supervisor never edits outcomes or computes SR. Seven
recovery-specific tests and the full evaluator suite passed; regenerated schema
was unchanged. No dependency or rendering workaround was applied.

At **22:21:54 UTC**, candidate coverage was **55/500** and Dense **28/500**, all
observed terminal outcomes successes. Both actual policy workers were alive in
that snapshot; this is a timestamped state, not a continuing liveness guarantee.
**Full SR remains null.** Evidence:
[executed coverage snapshot](evidence/visual-cache-spatial-20260919/progress-222154Z.json).
Supervisor logs remain in the full Spatial output root under
`refresh5-gpu1-recovery-01/` and `matched-recovery-01/`.

```bash
# From the frozen action-eval-visual-4701ac2 checkout, using the lane's original GPU.
PYTHONPATH="$PWD/src:$LIBERO_ROOT" CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python ../action-eval-recovery-f8eae67/scripts/recover_native_run.py \
  "$FROZEN_RUN" --evaluator-root "$PWD" --logs "$NEW_RECOVERY_LOG_DIR" \
  --max-attempts 20 --max-stagnant 3
```

Recovery command (same run directory, no configuration override):

```bash
PYTHONPATH="$PWD/src:$LIBERO_ROOT" CUDA_VISIBLE_DEVICES=0 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m action_eval.cli resume "$MATCHED_PILOT_RUN"
```

## Coverage safeguard

The original paired-SR script emitted rates on the completed CSV intersection and
used the wrong one-sided McNemar tail. Commit **5d3180f** fixes both. The revised
tool requires benchmark/protocol manifests, pairs seed as part of episode identity,
rejects duplicate or out-of-manifest rows, and leaves all rates/intervals/p-values
null unless both complete planned manifests have terminal outcomes. Missing/error
episodes cannot reduce the denominator. Eight regression tests passed on the server.
