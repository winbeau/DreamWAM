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
| 1–2 | Scope, authorized hardware, clean exact Git chain | Clean local/H100 `fda0235d3da9389c529b36204c66055020900b7a`; new commits use sync script |
| 3 | Paper/version audit, public implementation status | [Source audit](SOURCES.md); inference refinement separated from training |
| 4 | Typed raw-data manifest, licenses, frozen splits | Author data absent; 9 self-captured observations hash-verified; [plan](experiment-plan.json) freezes episode identities |
| 5 | Inherit all prior evidence | All 119 historical indexed artifacts and 9 observations verified, zero mismatches; legacy results remain immutable |
| 6 | Replayable multi-step/layer/head raw profile | Implement bounded native-attention capture; real-checkpoint capture pending GPU admission |
| 7 | Equal-budget interventions, action/video errors | Pending new profile and separated AV/VV interventions; semantic labels unavailable |
| 8 | Independent selectors and background pooling | Existing uniform/action/context retained; value/dynamic/pooling work pending |
| 9 | Bounded schedule scan, frozen choice | Existing 0/1-refresh negative/positive evidence retained; new selection-specific scan pending |
| 10 | CPU/CUDA/graph/checkpoint checks; complete paired timing | New checks pending; never inherit historical speed as a new-method result |
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
  new raw sources at `outputs/dido-sparse-profile-20260920/sources/` once archived.
- Limitations: this is provenance verification, not new accuracy/performance
  validation. Source-download failures through the web renderer were retried
  read-only with curl; no inference/error is counted as a task outcome.
- Next: bounded profile implementation and CPU tests on a detached H100 revision,
  while authorized GPU admission is unavailable.

## Inherited scientific controls

The prior 2.077× warm pilot used one full Dense anchor and nine **feature reuse**
steps, with uniform `[19,19,18]` visual read indices (56/294). It had no Sparse
step, so nominal Q10 was inactive. Both arms were 3/3; CPU-rendered whole
episodes were slower for the candidate. Repeated Dense was not independent data.
First-layer AV/context routing did not consistently beat uniform; structure-only
reuse reached 1.06–1.13×. All 999 timings, 153 interventions and 57 cases remain
in the inherited [report](../../action-eval/hybrid-routing-results-20260920.md).
