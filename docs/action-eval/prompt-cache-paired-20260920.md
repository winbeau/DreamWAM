# Matched closed-loop validation of the 1.872× guided sparse candidate

**Update at 2026-09-20 06:15 UTC:** the user reduced current screening to
**50 episodes per candidate** and deferred unified full tests until method
selection. The full Spatial controller and the legacy Long recovery were
stopped. Full Spatial Dense retains **129/500 successes**, and queued Sparse
did not start. The original manifests and accepted result hashes are preserved;
these incomplete full plans retain null SR. See the separate
[quick50 record](quick50-20260920.md). The timestamped launch history below
describes the earlier plan and is not an instruction to resume it.

Status at **2026-09-20 05:52:19 UTC: PILOT PAIR COMPLETE; full Spatial RUNNING;
official candidate SR null**. Both pilots finished **15/15 successes**, zero
task failures and zero error attempts. This is a plumbing/quality pilot, not
complete benchmark SR or a non-inferiority result.
The integrated candidate is the [measured guided visual-token path](prompt-cache-single-factor-20260920.md):
30/294 tokens refreshed at step 5 after a dense first-step anchor; eight visual
reuse steps; all ten action steps. Prompt encoding reuse is exact and enabled on
both Sparse and the strengthened conditioned-frame Dense control. Head × Stage
classification is not active in this candidate.

Frozen model: `04163905f047169a159b3e366d679afc86392757`.
Frozen evaluator/configs: `aa5f5fa7815702ee9dd00aedca2a9e94d0ff9133`.
Checkpoint SHA256:
`6c087e5b9e201f19dbe92834b470a7ee18937a18ea9adcbb49886aa1e3ac4c61`.
The real-checkpoint adapter audit and CPU/CUDA model tests passed before launch.
Four configs (two pilots and two independent full Spatial matrices) passed
`action-eval validate` and `doctor`; platform **199 tests passed in 112.55 s**,
and generated schema bytes match the committed schema. The evaluator
`pyproject.toml`/`uv.lock` hashes did not change, and no environment install or
sync was performed. MuJoCo **3.3.2**, robosuite **1.4.1**, numpy **1.26.4** were
verified by the actual simulator interpreter.

The pilot uses Spatial tasks **0/1/2 × initial states 0–4**, 15 episodes per arm,
with `dreamwam-release-v1`: max 400, wait 30, 256×256 cameras, replan 10, seed 42.
Model settings remain horizon 32, ten denoising steps, nine video frames, bf16,
and fixed-per-predict RNG. The full Spatial configs retain all **500** identities;
the former four-suite plan required **2,000/config** and is now deferred. Pilot counts cannot fill
the separate full-run denominators or establish acceptable SR loss.

## Placement and provenance

Both new pilot arms use policy GPU **2**
(`GPU-b15ccd2e-b17d-20a9-a130-b4b7eb37a074`) and dedicated renderer GPU **4**
(`GPU-490b4a76-6210-31b9-4e03-838a113cf5f4`). GPU 2 is shared with an existing
compute job; admission requires utilization ≤20%, at least 35,000 MiB free, no
graphics process, an empty renderer, and another available card left unused by
this task. The other task's rendering GPUs 0/6/7 remain untouched.

This is a **new matched rendering lineage**, kept separate from all preceding
frozen SR runs. It does not assume byte-equivalent rendering or equal SR to the
old placement. Both arms use the evaluator's existing native-read checker,
which calls the original read once and stops on an untouched RGB buffer.
The controller checks graphics-card processes every three seconds and stops
only its own run if another CUDA/graphics process enters that renderer. It also
checks the actual worker's visual/prompt options, sampling options and checkpoint
hash. Each pilot invocation has a 30-minute wall limit; there is no native-crash
retry loop in this controller. Shared-load episode timings are not used to
replace the separate 1.872× latency measurement.

## Actual attempts

- **05:28:44–05:28:51 UTC, FAILED startup, exit 1:** first Dense launch did not
  include the external LIBERO checkout on `PYTHONPATH`. It stopped before model
  startup, native reads or any episode outcome. No task failure was created.
  The original controller, launch log and zero-outcome directory are preserved.
- **05:31 UTC, VERIFIED import preflight:** added the existing LIBERO checkout
  to the launch environment, imported the real task suite/environment and
  checked simulator dependency versions. No dependency files were changed.
- **05:32:17–05:37:08 UTC, VERIFIED Dense pilot, exit 0:** controller **444011**,
  renderer/runner **444227**, **15/15 successes**, no task failures or errors.
  The native journal has **3,764** ordered read start/done pairs and zero errors.
- **05:37:50–05:42:34 UTC, VERIFIED Sparse pilot, exit 0:** renderer/runner
  **449935**, **15/15 successes**, no task failures or errors. Its native journal
  has **3,758** ordered read pairs and zero errors. Both actual worker fingerprints
  match the requested visual/prompt/sampling options and checkpoint.
- The independent pilot audit checked all 30 terminal records and their hashes,
  full 15-identity manifests, saved actions/inputs and complete native journals.
  All **15 initial proprio hashes match** between arms; **0/15 initial image pairs
  match bitwise**. This checks proprio, not the full hidden MuJoCo state. Rendering
  repeatability is still not established, even on the common graphics card.
  The outcome pairing has zero discordant cases in this small cohort; it cannot
  establish preserved official SR or rule out a loss on other episodes.
- **05:44:42 UTC, RUNNING full Spatial Dense:** new controller **453956**,
  renderer/runner **454021**, new 500-identity manifest. The **05:52:19 UTC** audit
  verifies **33/500 outcomes, all successes**, a live policy worker and the actual
  worker fingerprint. The same
  controller schedules the separate **500-identity Sparse run** afterward. Full
  runs use the validated `*-libero-spatial.yaml` files, the same GPUs and native
  checker, and a **12,000-second limit per invocation**. No pilot result is pooled
  into either full-run denominator.

Server evidence root: `outputs/prompt-cache-paired-20260920/`.
The corrected pilot pair is under `pilot-pair-v2/`; its controller records exact
commands, admission/monitor inventories, PIDs, exits and stop reasons. Local
[preflight evidence](evidence/prompt-cache-paired-20260920/preflight-aa5f5fa/)
contains every config validation/doctor result, platform tests, schema and the
explicit LIBERO import check. [Complete pilot artifacts](evidence/prompt-cache-paired-20260920/pilot-pair-v2/)
include the two actual runs, official terminal records, action traces, input
hashes and the independent `pilot-pair-audit.json` / `paired-pilot.json` reports.
The additional `pilot-trace-audit.json` verifies all 30 action traces: finite
32-action predictions and exactly their 10-action prefixes executed, truncated
only by the recorded terminal step. The original accepted result hashes match.
Pilot audit SHA256: `03eae027c16f5fcfce8dfb9ec2198b5dad2cd47cef0745cd1f3fa4036ae9d469`;
trace audit SHA256: `aa07b3cdd7d1189be3e819278a1681f7b7a6992eed3af28d13af955c921d209f`.
The [full Spatial snapshot](evidence/prompt-cache-paired-20260920/full-spatial/progress-20260920T055219Z.json)
preserves the new manifest and all 33 accepted outcome hashes. Live full-run
artifacts remain separate under `full-spatial/` on the server.

Command form from the frozen evaluator worktree, after setting `MODEL_ROOT`,
`LIBERO_ROOT`, `GPU_UUID` and `RENDER_GPU_UUID` to the recorded resources:

```bash
PYTHONPATH="$PWD/src:$LIBERO_ROOT" MUJOCO_GL=egl \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python scripts/supplement-libero.py \
  configs/experiments/dreamwam-prompt-conditioned-pilot.yaml \
  --render-gpu "$RENDER_GPU_UUID" --check-native-writes --output "$DENSE_PILOT"
```

The Sparse command uses `dreamwam-prompt-guided-pilot.yaml` and its own output
directory. The observed difference in policy options is visual computation;
the exact prompt cache and all scientific protocol settings match. Full official
episode pairing, SR uncertainty and the user's SR-tolerance decision remain pending.
