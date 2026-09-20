# DIDO-guided sparse-profile: isolated iteration chain

Created 2026-09-20 UTC. Setup/handoff only; no new GPU experiments started.

Execution amendment, 2026-09-20: the user has now invoked the saved goal. Follow
the current [progress ledger](PROGRESS.md) and AGENTS.md amendments: 50 total
closed-loop episode attempts across arms, SR tolerance 5 percentage points,
author raw data unavailable. The original setup statement above is historical.

## Repositories and scope

| Item | Location / identity |
|---|---|
| Local model worktree | `/home/winbeau/Papers/ICLR2027-WAM-SA/.trees/dido-sparse-profile` |
| Git branch | `experiment/dido-sparse-profile` |
| Git origin | `winbeau/DreamWAM` |
| Base revision | `ad0765b` (existing hybrid routing implementation/results) |
| H100 worktree | `/root/wenbiao_zhao/dreamwam-sr/.trees/dido-sparse-profile` |
| SSH host | `h100-server` |
| Model interpreter | `/root/wenbiao_zhao/dreamwam-sr/DreamWAM-fresh-6c52f36/.venv/bin/python` |
| Evaluator checkout | `/root/wenbiao_zhao/dreamwam-sr/action-eval` |
| Evaluator revision at setup | `5eab0bdd0f7e8bec200b3a4b12745e1a8030b773` |
| Evaluator interpreter | `/root/wenbiao_zhao/dreamwam-sr/action-eval-fresh-e0d9e80/.venv/bin/python` |
| LIBERO | `/root/wenbiao_zhao/dreamwam-sr/LIBERO` |
| LIBERO revision at setup | `8f1084e3132a39270c3a13ebe37270a43ece2a01` |
| Historical evidence | `/root/wenbiao_zhao/dreamwam-sr/outputs/hybrid-routing-profile-20260920` |

The enclosing local `.trees` directory is outside any Git repository; this is a
linked worktree of DreamWAM, not a new unrelated repository. Existing local main,
server main and historical run worktrees remain untouched. No action-eval source
changes are needed for setup. If evaluator/config changes become necessary, create
a separate action-eval worktree and branch rather than editing its main in place.

## Normal iteration

From the local worktree:

```bash
git status --short
git add <explicit-reviewed-files>
git commit -m '<bounded change and verification status>'
bash deployment/h100/sync-dido-worktree.sh
```

The sync script pushes the branch, fetches it on H100, creates the remote worktree
on its first run, executes `git pull --ff-only origin experiment/dido-sparse-profile`,
and requires server HEAD to equal the exact local commit. It rejects dirty source
trees, unexpected branches/paths and divergent history; it never stashes, resets,
force-pushes or overwrites assets. It links the existing checkpoint and Wan assets
into ignored directories. No package installation, model loading or GPU job occurs.

Do not pull while a run uses this worktree. Prefer a separate detached H100 worktree
under `.trees/dido-run-<commit>-<run-id>` for every live experiment. If using the
development checkout for a short run, maintain `.dido-live-run` for its lifetime;
the sync script refuses deployment while that marker exists (even if untracked).
The marker is only a guard, not automatic process discovery: check all owned jobs
before deployment. Never remove a live marker without verifying the process ended.

## Environment and execution

Use the existing verified environments unchanged. The new shell environment file
sets the **new** MODEL_ROOT and PYTHONPATH while reusing the known interpreter.
Do not source the old fresh-token environment alone: it points MODEL_ROOT at the
old `DreamWAM-fresh-6c52f36` checkout. The new file also keeps CPU OSMesa rendering.

On H100, after current resource admission:

```bash
export DREAMWAM_SR_ROOT=/root/wenbiao_zhao/dreamwam-sr
export GPU_UUID=<explicitly-admitted-GPU-UUID>
source /root/wenbiao_zhao/dreamwam-sr/.trees/dido-sparse-profile/deployment/h100/dido-sparse-profile.env.sh
cd "$MODEL_ROOT"
```

The file deliberately clears CUDA_VISIBLE_DEVICES. For a standalone profiler,
explicitly set `CUDA_VISIBLE_DEVICES="$GPU_UUID"` after admission; the evaluator
sets process placement itself. Verify `dreamwam.__file__` is from the intended
worktree before any real run. A frozen run must override MODEL_ROOT **and**
PYTHONPATH to the detached run worktree and record that revision.

Prior H100 authorization covers GPUs 3/4/5. GPU 3 was used previously; its recorded
UUID is `GPU-c0af33a9-498c-ff7c-bb56-e9992ccded30`. This is not a reservation:
recheck UUIDs, memory, utilization and processes before every launch; leave another
available authorized card unused, except for the user's latest explicit flexible
GPU 5 sharing authorization. GPU 0 was subsequently reclaimed; do not use it.
The profiler's GPU 5 exception requires at least 50,000 MiB free and utilization
at most 10% before model loading. It does not reserve the card or signal other
projects. No dummy-load reservation is run. CPU rendering has no render GPU. Do not use
other GPUs/hosts, revive historical queues, train the checkpoint, or alter the
scientific protocol without further authority.

Existing checkpoint SHA-256 (historical evidence, revalidate before experiments):
`6c087e5b9e201f19dbe92834b470a7ee18937a18ea9adcbb49886aa1e3ac4c61`.
Checkpoint and pretrained files remain external assets, never committed.

## Handoff and verification

Read [GOAL.md](GOAL.md) for the paste-ready future goal. Read the worktree AGENTS.md
and matching action-eval skills before adapter/config work. The skills live in
`../../action-eval/skills/` relative to this model worktree on either machine.
Read-only environment checks are not model accuracy or speed verification.
The setup evidence and tested source revision are recorded in
[SETUP-VERIFICATION.md](SETUP-VERIFICATION.md).
