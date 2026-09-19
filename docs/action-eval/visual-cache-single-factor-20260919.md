# Joint temporal visual reuse: speed passed, SR not established

Status: **MEASURED latency; official paired pilot in progress**. Date: 2026-09-19 UTC.
The complete goal (SR-constrained ≥1.5×, aiming for 2×) remains active and unproven.

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

## Coverage safeguard

The original paired-SR script emitted rates on the completed CSV intersection and
used the wrong one-sided McNemar tail. Commit **5d3180f** fixes both. The revised
tool requires benchmark/protocol manifests, pairs seed as part of episode identity,
rejects duplicate or out-of-manifest rows, and leaves all rates/intervals/p-values
null unless both complete planned manifests have terminal outcomes. Missing/error
episodes cannot reduce the denominator. Eight regression tests passed on the server.
