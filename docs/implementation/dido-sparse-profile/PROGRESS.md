# DIDO profile execution ledger

Status: ACTIVE, 2026-09-20 UTC. The full [GOAL](GOAL.md) is invoked; this is no
longer a setup-only handoff. No training or checkpoint changes are authorized.

User amendments: author raw data is unavailable to the user; SR tolerance is
5 **percentage points** below matched Dense; real closed-loop testing is capped
at 50 episodes in total across all arms and attempts. First proposed rollout is
3 pairs (6 episodes), only after profiling, implementation and adapter gates.
No rollout has started in this branch. A tiny pilot cannot establish statistical
non-inferiority at the 5-point margin.

## Requirement and evidence ledger

| GOAL section | Required evidence | Current state / next gate |
|---|---|---|
| 1–2 | Scope, authorized hardware, clean exact Git chain | All implementation stages committed/pushed/synced through `a2aa495`; server tests use separate detached worktrees |
| 3 | Paper/version audit, public implementation status | [Source audit](SOURCES.md); inference refinement separated from training |
| 4 | Typed raw-data manifest, licenses, frozen splits | Author data absent; 9 self-captured observations hash-verified; [plan](experiment-plan.json) freezes episode identities |
| 5 | Inherit all prior evidence | All 119 historical indexed artifacts and 9 observations verified, zero mismatches; legacy results remain immutable |
| 6 | Replayable multi-step/layer/head raw profile | Bounded native capture/NPZ replay implemented and CPU-tested; real-checkpoint capture pending GPU admission |
| 7 | Equal-budget interventions, action/video errors | Separate AV/VV/joint delete, zero-value replacement and recompute diagnostics CPU-tested; real input study and semantic labels missing |
| 8 | Independent selectors and background pooling | Score/selection and region-pooling reference modules CPU-tested; online hybrid integration, calibrated selection and graph validation pending |
| 9 | Bounded schedule scan, frozen choice | Existing 0/1-refresh negative/positive evidence retained; new selection-specific scan pending |
| 10 | CPU/CUDA/graph/checkpoint checks; complete paired timing | 33 CPU tests pass, one CUDA test skipped; CUDA/graph/checkpoint/timing gates remain open |
| 11 | Predeclared bounded paired LIBERO pilot | 0/50 episodes attempted; native OSMesa and adapter gates pending |
| 12 | Code/config/raw arrays/results/reproduction; release resources | Active ledger; no new method speed or SR claim; no owned GPU job |

## Initial verification

- Timestamp: 2026-09-20 17:33–17:44 UTC; source on both ends:
  `fda0235d3da9389c529b36204c66055020900b7a`.
- Checkpoint rehashed on H100:
  `6c087e5b9e201f19dbe92834b470a7ee18937a18ea9adcbb49886aa1e3ac4c61`.
- `git status --short --branch`, `git rev-parse HEAD`, `nvidia-smi`, process
  inventory and checkpoint `sha256sum`: exit 0. No task-owned live model job.
- Authorized GPU 3: 23,963 MiB / 100%; GPU 4: 24,249 MiB / 82%; GPU 5:
  4 MiB / 0%. GPU 5 is the last available authorized card and remains unused.
  These observations are not a reservation; recheck before any launch.
- Read-only SHA-256/size check over
  `docs/action-eval/hybrid-routing-evidence-20260920.json`: 119/119 artifact
  hashes match; nine trajectory observations also match; exit 0.
- Historical evidence-index SHA-256:
  `81db152ef9e91599b3d98e9c731fdf36e27b4ae345821199be9d0482d4b73b5b`.
- Input manifest SHA-256:
  `6b1882ab8ac01366fbe6ec2e4463ceb450adb5c7276f6d2ab613ef72128cc66f`.
- Artifacts: existing H100 `outputs/hybrid-routing-profile-20260920/`;
  new raw sources archived at `outputs/dido-sparse-profile-20260920/sources/`,
  all six source-file hashes revalidated after transfer.
- Limitations: this is provenance verification, not new accuracy/performance
  validation. Source-download failures through the web renderer were retried
  read-only with curl; no inference/error is counted as a task outcome.
- Next at this initial audit: bounded profile implementation and CPU tests on a
  detached H100 revision. Results are recorded below.

## Implemented and verified through 18:01 UTC

Implementation source: `a2aa495a84c4a2f0883c461078a6b625ba5fdfba`.
H100 detached worktree: `.trees/dido-run-a2aa495-profile-verification`.
Latest checks ran 2026-09-20 **18:01:36–18:01:42 UTC**, exit 0:
**33 passed, 1 skipped** (CUDA-only). CPU tests use a small synthetic Joint model;
they are not real-checkpoint, GPU speed or control evidence. The CLI import/help
check also exits 0. Earlier stage checks (21+10 passes) are retained, not counted
as additional scientific samples. Full commands/hashes are in
[VERIFICATION.json](VERIFICATION.json).

The implemented [profiler](PROFILE.md) stores sampled native Q/K/V, full joint
mask/probabilities, all ten denoising steps, frame/camera indices, original RoPE,
value proxies and two separate time axes. The real runner requires exact parity
of executable actions, **pre-binarization raw actions and video latents** against
an independent output-only native control. That real run is still unstarted.

Reference interventions keep dense tensor sizes/work and independently target
AV or VV. CPU checks confirm that an AV-only intervention preserves the video
trajectory, whereas a last-layer/last-step VV-only intervention preserves the
action while changing video. These are controlled synthetic implementation
checks, not a finding about real task importance. Full-recompute diagnostics
degenerate to native output, and old request state is discarded.

Reference pooling averages **post-RoPE K** and unrotated V, retaining original
query positions. `count` and `unit` multiplicity are separate options. It rejects
mixed visibility, cross-camera/frame groups and unlabelled mixed cache ages.
Preserving 98 current-frame cells leaves a minimum of 162 packed visual rows
under 2×2 future-background pooling; compare against hard selection at the same
actual row count, not the historical 56-row candidate. It is not yet wired into
the accelerated runtime. No combination weights have been calibrated or adopted.

At **17:58:53 UTC**, authorized GPU 3/4 remained at 100% utilization with
23,963/24,249 MiB used; GPU 5 had 949 MiB and was the last available authorized
card. No task-owned model process was live. New checkpoint calls: **0**;
closed-loop episode attempts: **0/50**. No GPU wait is described as a live run.

This goal turn is **progress** (committed implementation, exact deployment,
artifact verification and server tests), not completion or a verified job wait.
The goal remains active. Next: finish raw-profile analysis/reporting, recheck
GPU admission, capture the 9-input native profile and use its real evidence to
choose bounded interventions before online selector integration. Remaining
runtime, refresh-scan, timing, adapter and finite SR deliverables retain the full
original scope.

## Inherited scientific controls

The prior 2.077× warm pilot used one full Dense anchor and nine **feature reuse**
steps, with uniform `[19,19,18]` visual read indices (56/294). It had no Sparse
step, so nominal Q10 was inactive. Both arms were 3/3; CPU-rendered whole
episodes were slower for the candidate. Repeated Dense was not independent data.
First-layer AV/context routing did not consistently beat uniform; structure-only
reuse reached 1.06–1.13×. All 999 timings, 153 interventions and 57 cases remain
in the inherited [report](../../action-eval/hybrid-routing-results-20260920.md).
