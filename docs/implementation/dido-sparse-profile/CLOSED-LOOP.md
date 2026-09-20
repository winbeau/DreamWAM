# Bounded paired rollout controls

Status: launch support passes 17 H100 CPU tests and the evaluator-interpreter
CLI check at `72816b1`, 2026-09-20 19:48:40–19:48:41 UTC, exit 0. Logs and hashes
are in [ONLINE-VERIFICATION.json](ONLINE-VERIFICATION.json). No DIDO
closed-loop episode or real effort-ledger reservation has been started.
The first pilot remains three matched Spatial episodes per arm: tasks 0/1/2,
initial state 1, seed 42. The protocol remains `dreamwam-release-v1`, with CPU
OSMesa rendering, one worker, retries=0 and error_policy=stop. Only the evaluator
defines success. The unchanged offline, adapter and native-rendering gates in
[experiment-plan.json](experiment-plan.json) remain prerequisites.

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
`--max-shared-utilization 50` only for the frozen adapter/six-attempt pilot under
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

## Later invocation

After the candidate/config and real adapter checks are frozen, the existing
paired launcher receives the following additional controls:

```
--planned-episodes 3 --first-arm dense
--authorized-gpus 3 4 5 --render-backend osmesa --share-gpu5
--episode-ledger /root/wenbiao_zhao/dreamwam-sr/outputs/dido-sparse-profile-20260920/closed-loop-ledger.json
```

The required evaluator/model roots, two actual config paths, verified adapter
report, fresh output directory and finite wall limits must name the final frozen
run. This is a flag specification, not a fabricated runnable experiment with an
unselected candidate. Configuration changes belong in an isolated action-eval
branch/worktree following its skills. No SR or non-inferiority claim follows
from these launch controls; the user permits at most a 5-percentage-point drop
against contemporaneous matched Dense, and the small sample's interval remains
part of the report.
