# M1 whole-chunk allocation and readable execution evidence

Status at 2026-09-21 15:28 UTC: implementation, two five-pair pilots, 725 matched
timings and final trace galleries COMPLETE; Desktop export pending. This continues the user's request to implement
M1/M2/M3 and make the figures understandable. Historical pilot results remain
Dense 5/5 versus M1–M3 3/5; the previous 50-pair/full-suite queues stay deferred.

## Change and controls

The old M1 changed per-refresh Q/KV quotas, but adaptive M3 could consume the same
extra full-pass budget at every M1 level. Smaller refreshes could therefore mean
more refreshes, rather than a smaller chunk allocation. The opt-in
`chunk_extra_passes` now gives M1 an enforceable whole-chunk visual Q-row cap:

`cap = min(steps × N, N + ceil(chunk_extra_passes × N))`.

The mandatory Dense first step counts toward the cap. For the real 294-token
grid, low/medium/high allocations of 0.1/0.2/0.3 extra passes permit
324/353/383 total rows, including 294 anchor rows. These are per-sample visual
query-row budgets, not FLOP or latency guarantees; action work, key reads and
attention probes remain separate counters. Actual spending and unused allocation
are exposed in adapter diagnostics. The fixed M3 control rejects an infeasible
allocation before transformer execution. Adaptive M3 cannot overspend and skips
drift/age probes once no further refresh is affordable.

Frozen configurations:

- `configs/sparse/m1-total-budget-fixed-m2-m3.json`: unchanged AV→VV selector,
  Dense step 0 / Sparse step 5 / Reuse elsewhere, with explicit M1 chunk caps.
- `configs/sparse/m123-total-budget.json`: the same M1 caps and unchanged adaptive
  M3 drift/age decision rules. Only the cap interface and exhausted-budget probe
  shortcut change relative to the earlier adaptive candidate.
- `configs/sparse/m123-total-budget-kv75.json`: a separate available read-floor
  control. Its additional five-pair run is labelled separately; it must not be
  combined with the cap-factor pilot or presented as evidence for the cap alone.

Raw-observation features, calibrated thresholds and histories are unchanged.
These thresholds are development heuristics, not demonstrated task-phase or
sparsification-sensitivity estimates. Training and the scientific protocol are
unchanged. Reuse occurs within each chunk; all visual cache state clears between
chunks. Every chunk still encodes its own real first-frame RGB.

## Source, environment and checks

Local commit/push → clean server fast-forward pull → detached worktree was used
throughout. Model change: `d566382`; trace/visibility renderer: `1a32c29`;
pilot/initial timing tooling: `04b0cb4`; timing resource audit: `6fdf970`.
Timing cleanup correction: `0055544`. The first 464-request timing attempt
`timing-6fdf970` passed its per-call checks but exited 1 during duplicate wrapper
cleanup; its final summary was not saved and the attempt is not labelled complete.
Original journals and traceback are retained; a clean rerun is required.
Worktrees are under H100 `/root/wenbiao_zhao/dreamwam-sr/.trees/`.
The Python 3.10.20 / torch 2.7.1+cu126 environment is reused unchanged.
Checkpoint SHA256 remains
`6c087e5b9e201f19dbe92834b470a7ee18937a18ea9adcbb49886aa1e3ac4c61`.
No dependency installation or lockfile change occurred.

Full model/source identities are `d566382be8cadf864cd6f5036cecbf2a04d07873`
(core), `04b0cb46c43d4d7b50117f2790fc2f63e34c5fc8` (cap pilot),
`96492288ef21dcb467d180007a2cdff6dba1c932` (KV75 pilot, final timing/captures).
Final rendering uses `626c7ae`; every `render.json` records its full commit and
script hash. Source hashes for pyproject, requirements and H100 model lock remain
`2d717d498ffd31e612a6b523ee4428b8788bd7b7f9f804c33776114f64afafac`,
`22f662d3eb88634277dea66d81114368eb9f15ce4a5c47e21f860bcd9ffd4448`, and
`435692180f4d5a28e47d00be00cba35ea83cca8f9cdbd2566b96eec819098354`.

All new artifacts are under H100
`outputs/m1-m2-m3-20260921/total-budget-v2/` (`OUT` below).

| Verification | Source | Result / exit code | Artifact |
|---|---|---|---|
| CPU sparse suite | d566382 | 399 passed, 23 CUDA skips, 16 subtests / 0 | cpu-d566382.log |
| CUDA sparse suite | d566382 | 422 passed, 16 subtests / 0 | cuda-d566382.log |
| Trace/visibility/missing-value tests | 04b0cb4 | 7 passed / 0 | trace-tests-04b0cb4.log |
| Fixed M2/M3 real adapter | d566382 | 116 calls, own uncached eager bitwise / 0 | fixed-adapter-d566382/report.json |
| Adaptive M3 real adapter | 04b0cb4 | 116 calls, own uncached eager bitwise / 0 | adaptive-adapter-04b0cb4/report.json |
| KV75 real adapter | 9649228 | 116 calls, own uncached eager bitwise / 0 | kv75-adapter-9649228/report.json |
| Both pilot configs validate/doctor | 04b0cb4 | passed / 0 | *.validate.log, *.doctor.log |
| Native OSMesa describe | 04b0cb4 | expected RGB/action behavior / 0 | native-describe.log |

Commands use the unchanged reference interpreter. Fixture tests do not load the
checkpoint; adapter checks do. The latter run
`scripts/sparse/verify_chunk_adapter.py --inputs <observations-dev3/manifest.json>
--options <configuration above> --out-dir <fixed/adaptive adapter directory>`.
They cover complete causal histories, own-reference actions, work/route counts,
budget decisions, episode reset, stable fingerprints and cache release. Their
setup-inclusive durations are not latency measurements.

The additional `budget-and-frozen-control-audit.json` verifies all nine previously
captured fixed-M3 actions bitwise against the new fixed-control replay. Across
29 complete-history adaptive calls, actual low/medium/high spending is exactly
324/353/383 rows (9,720/10,590/11,490 visual token-layers), within each allocation.
It reads only archived arrays and adapter diagnostics, without a new model run.

## GPU placement

The user has authorized H100 0 and negotiated 3–5; H200 remains paused. On the
fresh check, GPU 0 had only approximately 8 GiB free. The attempted renewal of
the owned GPU-3/4 holders at 14:19 UTC failed their empty-card checks during
handoff; these attempts are preserved, and neither was assumed to be holding.
GPU 4 subsequently acquired an external job and was not signalled.

At 14:29:44 UTC, empty GPU 3 was successfully reserved with a bounded 100-minute
holder (PID 1403541, token/start-ticks verified). That holder was stopped through
its ownership-checking interface immediately before CUDA tests. Current model
placement is GPU 3, UUID `GPU-c0af33a9-498c-ff7c-bb56-e9992ccded30`.
GPU 5 remains unused by this effort. The host is shared; an empty policy card
does not imply an exclusive host. Timing checks also account for allocations
hidden from container-visible process lists.

## Figures

`trace_explainer.py` reads hash-verified captures into a separate directory. It
adds a four-panel observed-frame guide (RGB/token IDs → AV → VV → actual reads
and updates), explicitly numbered rows/columns, and a chunk-budget ledger.
Absent executed keys are gray and hatched; valid low-probability keys remain
dark. New captures store the native VV visibility mask so structurally forbidden
associations can be hatched separately. Old numerical zeros are never treated as
proof of a structural mask. Future latent rows are labelled as having no observed
RGB. The first Dense step naturally updates every patch.

All AV summaries average heads/action queries while preserving joint [V,A]
softmax normalization. VV uses weighted seeds and its own color scale. The
fused kernel does not expose probabilities; these are reconstructions from its
actual Q/K inputs. Patch cells denote nominal coordinates, not isolated VAE
receptive fields or causal importance.

The current compact selector distributes query quotas across all three frames.
The conditioned first frame can have zero drift, so its tied query scores select
early token IDs by stable order. A cluster of green plus signs near the top left
therefore does **not** establish visual importance. This is a limitation of the
frozen M2 control, preserved rather than silently changed during the M1 factor.

Rendering uses the existing Noto Sans CJK font copied as an external data asset,
with its license, without environment installation. Font SHA256:
`b76b0433203017ca80401b2ee0dd69350349871c4b19d504c34dbdd80541690a`.
The initial Chinese example was visually inspected for readable glyphs, axes,
token IDs and legends. Fresh captures and Desktop export are pending below.

The initial new capture at `04b0cb4` completed 29 causal inputs with bitwise
uninstrumented actions, saving nine chunks and 171 research PNG/PDF pairs; its
Chinese renderer saved 117 PNG/PDF pairs. Fixed screenshot indices 0/1/5/9 can
miss an adaptive Sparse step. The final capture tooling (`9843c1f`, with indexed
reading aids at `ade3ed3`) therefore adds `--steps routed`: it observes the
uninstrumented reference schedule and captures the anchor, actual non-Reuse
steps and final step. This changes instrumentation coverage, never routing.
`--gallery none` separates tensor capture from CPU plotting. These final capture
and export checks remain pending at this snapshot.

## Measurement setup

`benchmark_chunk_sequences.py` compares Dense, the previous adaptive candidate,
the new fixed-M3 control and the new adaptive candidate on the same 29
complete-history inputs. The follow-up also includes the isolated KV75 read-floor
control, with five cyclically rotated repetitions balancing all five order positions.
Graph setup, eager references and first instruction misses are separated from
warm timing. Every measured action and causal decision must match its own
uncached eager replay; any new graph capture invalidates a timed request.

The new pilot preserves `dreamwam-release-v1`: Spatial tasks 0–4, init 0, seed 42,
400-step limit, 30 wait steps, replan every 10, two 256-square cameras. It uses
the pinned evaluator `e5a5cc5e05d381b0cc5bafbe4e000343c4b6e29f`, CPU OSMesa,
robosuite 1.4.1 and MuJoCo 3.3.2. Only five matched episodes per arm are planned.
Complete benchmark-owned outcomes and initial RGB/state pairing are required
before reporting pilot SR. Neither timing nor implementation parity establishes
quality preservation. Next: finish and append timing, trace and pilot evidence.

## Completed cap-factor pilot, 2026-09-21 15:09 UTC

**PILOT_PAIR_COMPLETE**, exit 0, source
`04b0cb46c43d4d7b50117f2790fc2f63e34c5fc8`, evaluator and checkpoint as above.
Controller started at 15:00:27 UTC; Dense ran 15:00:31–15:03:35 UTC and the
candidate started at 15:03:44 UTC. All ten episodes reached native terminal
outcomes without errors or forced cleanup. Complete rates: **Dense 5/5 versus
candidate 3/5**. Candidate tasks 0/2/3 succeed and tasks 1/4 reach the 400-step
limit. The task-3 recovery and task-4 regression relative to the previous pilot
are descriptive observations across separately executed cohorts, not paired
evidence that the cap caused either change.

Initial RGB from both cameras and robot state match bitwise on all five pairs.
Wilson intervals remain 56.6–100% for Dense and 23.1–88.2% for the candidate;
task-stratified bootstrap is withheld with one initial state per task. This is
another negative development pilot, not evidence of SR preservation.

Launcher command under the frozen evaluator environment:

```sh
python scripts/sparse/run_fresh_token_pair.py --eval-root "$EVAL_ROOT" --model-root "$MODEL_ROOT" --out-dir "$OUT/pilot5" --adapter-report "$OUT/adaptive-adapter-04b0cb4/report.json" --dense-config "$MODEL_ROOT/configs/action_eval/m123-dense-pilot5-osmesa.yaml" --sparse-config "$MODEL_ROOT/configs/action_eval/m123-total-budget-pilot5-osmesa.yaml" --planned-episodes 5 --render-backend osmesa --first-arm dense --authorized-gpus 0 3 4 5 --wall-seconds 1800 --admission-seconds 120 --stop-grace-seconds 180
```

`MODEL_ROOT` is `.trees/m123-total-04b0cb4`; source the committed
`deployment/h100/fresh-token-osmesa.env.sh` with the H100 root and GPU-3 UUID,
then override model/evaluator paths to the frozen worktrees. Admission records
explicitly show an owned completed-capture process retaining 13,767 MiB while
plotting, at 0% GPU utilization; GPU 5 was left unused. The host and initial
same-device allocation are therefore disclosed as shared. No policy process from
another user was signalled. Main source/environment hashes are unchanged.

At 15:09 UTC, `paired_sr.py`, `summarize_policy_timings.py` and
`audit_initial_inputs.py` each exited 0 for `<OUT>/pilot5/{dense,sparse}/run`.
Results are `paired-sr.json`, `pilot-timings.json` and `initial-input-audit.json`.
Pilot warm durations include any first encounter with a new graph shape, and the
two arms visit different states; the separate matched-input rerun is required
for a steady inference-speed result. Next: the KV75 single-factor diagnostic and
clean timing/capture completion, with no 50-pair or historical queue restart.

## KV75 read-floor diagnostic, completed 2026-09-21 15:19:21 UTC

**PILOT_PAIR_COMPLETE**, exit 0, source `9649228`, same evaluator/checkpoint/
protocol. This changes the read budget from 25/50/75% to 75/75/75%, retaining
the same query quotas, total caps, raw-observation controller, selector and M3
rules. Both configs pass validate/doctor and native describe; 116 adapter
verification calls pass own uncached eager parity. No weights or protocol change.
GPU 3 was empty at admission, and GPUs 4/5 were recorded as unused spares.
Later GPU-4 trace capture shared the host; closed-loop timings are descriptive.

Dense ran **15:12:15–15:15:17 UTC** and the candidate ran
**15:15:18–15:19:21 UTC**. Both achieve **5/5**, with zero errors or forced
cleanup. All initial RGB/state hashes match within the pair and across the
cap-only versus KV75 candidates. These are five development cases, not an SR
preservation claim or a complete benchmark.

| Task | Dense executed steps | Cap-only candidate | KV75 candidate |
|---|---:|---|---:|
| 0 | 73 | success | 76, success |
| 1 | 107 | failure at 400 | 112, success |
| 2 | 92 | success | 99, success |
| 3 | 80 | success | 88, success |
| 4 | 117 | failure at 400 | 367, success |

The drawer task remains slow: 367 steps versus 117, close to the 400-step limit.
Mean whole-episode time is **27.975 s Dense versus 39.577 s KV75**. Faster model
calls therefore do not imply faster task completion. The separate cap-only
cohort is 28.290 s versus 49.687 s. Preserve both outcomes rather than replacing
the negative pilot with the successful count of a different configuration.

Command: the cap-pilot launcher command above, with model worktree
`.trees/m123-kv75-9649228`, output `pilot5-kv75`, adapter report
`kv75-adapter-9649228/report.json`, and candidate YAML
`configs/action_eval/m123-total-budget-kv75-pilot5-osmesa.yaml`.
Postprocessing scripts above exited 0 at approximately 15:21 UTC; artifacts are
`kv75-{paired-sr,pilot-timings,initial-input-audit}.json` and
`cap-vs-kv75-input-audit.json`. The next candidate should retain this read-floor
control while investigating the drawer trajectory, without scaling the cohort yet.

## Matched-input latency, completed 2026-09-21 15:25:00 UTC

**COMPLETE**, exit 0, source `9649228`, checkpoint/environment unchanged.
Run **15:20:30–15:25:00 UTC**, physical H100 GPU 3, 29 complete-history inputs,
five variants × five rotated repetitions = **725 full adapter predictions**.
Each variant contributes 130 warm calls and 15 episode-first prompt misses.
Every prediction and causal budget/route decision matches its own uncached eager
reference. Graph capture is excluded and graph identity stays unchanged within
every timed prediction. All 77 resource checks find no foreign policy-GPU
process and at most 9 MiB of unexplained memory. This is an exclusively occupied
policy GPU on a shared host, not an exclusive machine.

| Variant | Warm mean ms | Dense / candidate |
|---|---:|---:|
| Strengthened Dense | 263.717 | 1.000× |
| Previous adaptive M1–M3 | 196.829 | 1.340× |
| New M1, fixed M2/M3 | 144.078 | 1.830× |
| New total-budget M1–M3 | 147.842 | 1.784× |
| New total-budget M1–M3, KV75 | 148.936 | 1.771× |

Dense uses the same exact prompt cache, conditioned-frame reuse and transformer
graphs. The KV75 read floor costs approximately **1.094 ms** over the new
adaptive candidate in this input cohort. Including episode-first prompt misses
with already warmed graphs gives 265.525 ms Dense versus 150.854 ms KV75;
model loading and graph capture remain excluded. This does not supersede or pool
the earlier shared-load GPU-0 timings. The fixed-M3 control is slightly faster;
adaptive M3 has not demonstrated an incremental speed benefit over it.

```sh
CUDA_VISIBLE_DEVICES=GPU-c0af33a9-498c-ff7c-bb56-e9992ccded30 OMP_NUM_THREADS=1 HF_ENDPOINT=https://hf-mirror.com timeout 1200 "$MODEL_PYTHON" scripts/sparse/benchmark_chunk_sequences.py --inputs "$INPUTS/manifest.json" --options configs/sparse/m123-calibrated.json --options configs/sparse/m1-total-budget-fixed-m2-m3.json --options configs/sparse/m123-total-budget.json --options configs/sparse/m123-total-budget-kv75.json --reps 5 --authorized-gpus 0 3 4 5 --out-dir "$OUT/timing-9649228"
```

`INPUTS` is the sibling `observations-dev3` directory. Raw timings, exact options,
reference actions, eager parity and resource snapshots are retained in
`timing-9649228/{requests.jsonl,report.json,references.npz}`. The earlier
`timing-6fdf970` cleanup failure remains archived separately and is not pooled.

## Final captures and remaining limits

Both cap-only and KV75 captures use `9649228`, on authorized empty GPU 4 while
GPU 3 runs the independent pilot. GPU 5 is left unused. The first KV75 capture
admission attempt exits before model execution when spare utilization changes;
the bounded resource wait admits the later attempt. Admission logs are retained.
Each variant replays all 29 inputs, verifies bitwise uninstrumented actions and
saves nine actual chunks. Both capture reports finish
`VERIFIED_REAL_CHECKPOINT_TRACE`, exit 0. Command:

```sh
CUDA_VISIBLE_DEVICES=GPU-3af22086-0ac4-4277-1486-96909b440d1c OMP_NUM_THREADS=1 HF_ENDPOINT=https://hf-mirror.com timeout 900 "$MODEL_PYTHON" scripts/sparse/capture_hybrid_trace.py --inputs "$INPUTS/manifest.json" --options configs/sparse/m123-total-budget.json --out-dir "$OUT/routed-capture-9649228" --layers 0,29 --steps routed --gallery none
```

The KV75 command substitutes `m123-total-budget-kv75.json` and
`kv75-capture-9649228`. CPU rendering from frozen `626c7ae`:

```sh
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 "$MODEL_PYTHON" scripts/sparse/trace_explainer.py --source "$OUT/routed-capture-9649228" --out-dir "$OUT/read-this-first" --font "$DREAMWAM_SR_ROOT/assets/fonts/NotoSansCJK-Regular.ttc"
```

For KV75, substitute the matching source and `read-this-first-kv75` destination.
Both render jobs exit 0 with **90 PNG + 90 PDF each**, index pages, token maps,
source/renderer/font hashes. The low-budget KV75 four-panel and nine-panel
examples were visually inspected, including Chinese labels and missing-value
hatching. The separate [Chinese reading guide](m123-reading-v2-zh.md) explains
outer rows/columns versus inner patch coordinates.

In all 29 development calls, M3 chooses **D0, R1–2, S3, R4–9**. This demonstrates
budget enforcement and working route execution, not varying task-phase-aware
timing. The M1 signal remains a raw-observation activity proxy; the current M2
first-layer AV/VV proxy and frame-balanced query quotas have not demonstrated
causal token importance or superiority over uniform/random controls. Further
M1/M2 ablations and the slow drawer trajectory remain substantive work. The
current implementation is usable and traceable; broader quality preservation is
unestablished, and the 50-pair queue remains deferred.

After all GPU measurements finished, owned GPU 3 was returned to a bounded
40-minute holder at **15:28:02 UTC**, ending approximately **16:08:02 UTC**.
GPU 4 was empty and left unused at admission. State file:
`gpu-hold/m123-v2-final-20260921/gpu3.json`; PID 1470693, token
`30c67116ee3fa7a0959c541a3dbd6cdc`. The ownership-checked status is `HOLDING`;
startup/status/admission snapshots are retained under `OUT/final-*`.
Only the recorded state/token/start-ticks may be used to stop it; a bare PID is
not sufficient authority. No evaluation or historical queue remains running.
