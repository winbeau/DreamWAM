# Bounded paired rollout controls

Status: both finite cohorts completed and independently audited, then stopped at
12 actual attempts / 12 charged slots. Development tasks 0/1/2 × init 1 give
Dense/candidate 3/3 each; fixed tasks 3/4/5 × init 2 give Dense 2/3, candidate
3/3. There are no errors/retries; Dense task 5 is a real 400-step failure. See
[REPORT.md](REPORT.md) and [FINAL-VERIFICATION.json](FINAL-VERIFICATION.json).
The five-point SR margin is not established by these tiny cohorts.

Launch support passed 17 H100 CPU tests and the evaluator-interpreter CLI at
`72816b1`, then 15 admission/ledger checks at controller `b847775`. The final
22-call adapter check, native renderer and config validation passed before
rollouts. Both cohorts retain `dreamwam-release-v1`, seed 42, CPU OSMesa, one
worker, retries=0 and error_policy=stop. Only the evaluator defines success.

## GPU 5 sharing

`scripts/sparse/run_fresh_token_pair.py --share-gpu5` implements the user's
explicit exception to leaving a second card unused. It only accepts physical
GPU 5, explicit paired configurations and CPU OSMesa rendering. Before each arm,
it requires at least 50,000 MiB free and utilization at most 10%, matching the
native-profile admission limits. It does not reserve the GPU, start dummy work,
or signal another project's processes. GPU 0 remains unauthorized. The original
launcher behavior is preserved when this new option is absent.

After the completed nine-input screen, the GPU remained near 40% utilization
with more than 76000 MiB free. The user-facing operational amendment uses
`--max-shared-utilization 50` only for the frozen adapter/two six-attempt cohorts under
the existing flexible-sharing authorization. The default remains 10; 50 requires
`--share-gpu5`, retains the 50000-MiB memory floor and is recorded in controller
metadata with its own source hash. Profiling/timing gates are unchanged. Shared
execution can increase both projects' latency and is not exclusive-hardware timing.

## One 50-episode effort ledger

Every invocation for this effort supplies the same `--episode-ledger` path:

```
/root/wenbiao_zhao/dreamwam-sr/outputs/dido-sparse-profile-20260920/closed-loop-ledger.json
```

`EpisodeBudget` serializes reservations with an OS file lock and persists them
with an atomic replacement and fsync. Before either arm's evaluator process
starts, the whole pair is charged: a three-pair pilot reserves six slots. Every
subsequent invocation checks the same immutable cap of 50 across all arms.
Concurrent invocations cannot oversubscribe the budget. Duplicate reservations,
changed effort IDs/caps and corrupt ledgers are errors, never reasons to start
with a new empty history.

Charges are conservative upper bounds. A refusal before the first process
reservation charges nothing. After reservation, an error or interruption retains
the whole charge, including any unstarted second arm; there is no automatic
refund or retry. This can leave fewer than 50 actual attempts, while ensuring
the effort cannot exceed 50 through repeated failed runs. The controller reports
the number of terminal attempt records separately, includes errors, excludes
explicit `not_run`/`not_attempted` records, and archives result hashes. Incomplete
records are not proof that an unrecorded attempt never started. A lock file or
reservation also does not prove a process is alive.

## Preserved launch specification

The completed development pair used the following additional controls; the fixed
pair used `--first-arm sparse`. Both explicitly set `--max-shared-utilization 50`:

```
--planned-episodes 3 --first-arm dense
--authorized-gpus 3 4 5 --render-backend osmesa --share-gpu5
--episode-ledger /root/wenbiao_zhao/dreamwam-sr/outputs/dido-sparse-profile-20260920/closed-loop-ledger.json
```

The model was `41f515a`, controller `b847775`, development evaluator `0f856c9`
and fixed evaluator `b5952a5`. Both used 120-second admission, 600-second arm
limits and 120-second stop grace, under an outer 25-minute timeout. Exact arm
commands, configs, GPU snapshots and fingerprints remain in their controller
artifacts. No runtime checkout was pulled while in use.

Both ledger reservations are finalized, with six actual attempts each. Do not
fill the remaining 38 slots or rerun these cohorts automatically. CPU evidence
replay in [REPORT.md](REPORT.md) requires no model or new episode. A later
experiment needs a newly scoped user request; preserve this original ledger and
all accepted outcomes.
