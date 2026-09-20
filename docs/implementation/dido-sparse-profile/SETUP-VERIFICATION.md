# Isolated worktree setup verification

Status: **VERIFIED setup chain**, 2026-09-20T17:28:03Z.
Tested implementation revision: `df6bafadbb32f21fea3aa6aa21bf6be04591d0f2`.
The subsequent verification-record commit changes documentation only and uses
the same sync script to exercise a second, fast-forward deployment.

## Checks completed before deployment

- Local DreamWAM main was clean at `ad0765b`; created the separate branch
  `experiment/dido-sparse-profile` and worktree `.trees/dido-sparse-profile`.
- SSH access to `h100-server` succeeded; the existing H100 hybrid development
  checkout was clean at `ad0765b`. No existing checkout was reset or switched.
- Existing checkpoint, Wan pretrained assets, model/evaluator interpreters,
  LIBERO checkout and private OSMesa library directory were found. No install,
  model load, profiling run or GPU job was performed.
- Created the guarded Git/SSH synchronization script, H100 environment entrypoint,
  workflow runbook and detailed future goal prompt. Model runtime is unchanged.

## Completed commands and outcomes

1. Local `git diff --cached --check`: exit 0. Created and committed six scoped
   files (branch instructions, deployment scripts and handoff documents).
2. Local `bash deployment/h100/sync-dido-worktree.sh`: exit 0. Pushed the branch;
   H100 fetched it, created the corresponding tracking worktree, and executed
   `git pull --ff-only origin experiment/dido-sparse-profile`. Local HEAD,
   local tracking ref, H100 HEAD and H100 tracking ref all matched the tested SHA.
3. H100 `bash -n` on both added shell scripts: exit 0.
4. H100 sourced the new environment file and explicitly set
   `CUDA_VISIBLE_DEVICES=''` for **CPU-only** import checks. With MODEL_PYTHON,
   imported torch, dreamwam and action_eval_sdk; asserted that dreamwam's source
   belonged to this new worktree and CUDA remained uninitialized. With EVAL_PYTHON,
   imported action_eval and action_eval_sdk. The combined check exited 0.
5. Asset-link resolution and file/directory existence checks passed. Both
   worktrees remained clean. Original local main stayed at
   `ad0765b6120b80b8d92620f15c3d90afce7cb614`; action-eval remained unchanged.

Observed verification output:

```text
DIDO_SYNC_OK branch=experiment/dido-sparse-profile sha=df6bafadbb32f21fea3aa6aa21bf6be04591d0f2 worktree=/root/wenbiao_zhao/dreamwam-sr/.trees/dido-sparse-profile
CPU_ONLY_MODEL_IMPORT_OK 3.10.20 2.7.1+cu126
dreamwam_source /root/wenbiao_zhao/dreamwam-sr/.trees/dido-sparse-profile/dreamwam/__init__.py
sdk_source /root/wenbiao_zhao/dreamwam-sr/action-eval/packages/policy-sdk/src/action_eval_sdk/__init__.py
cuda_initialized False
```

Verified import/asset locations:

- Model: `/root/wenbiao_zhao/dreamwam-sr/.trees/dido-sparse-profile/dreamwam/__init__.py`.
- Evaluator: `/root/wenbiao_zhao/dreamwam-sr/action-eval/src/action_eval/__init__.py`.
- SDK: `/root/wenbiao_zhao/dreamwam-sr/action-eval/packages/policy-sdk/src/action_eval_sdk/__init__.py`.
- Checkpoint: `/root/wenbiao_zhao/dreamwam-sr/assets/DreamWAM/checkpoints/dreamwam_joint.pt`.
- Wan assets: `/root/wenbiao_zhao/dreamwam-sr/assets/DreamWAM/pretrained/Wan2.2-TI2V-5B`.

## Limitations and next action

No packages were installed/upgraded; no checkpoint was loaded, no CUDA context
was initialized, and no profiling, rollout or training was started. This does not
verify model numerical parity, native rendering, current checkpoint content hashes,
available GPU capacity, speed or SR. The goal prompt requires fresh admission and
those experiment-specific gates before use. No author raw dataset was acquired.

Next action belongs to the user/new session: enter the new local worktree and
invoke [GOAL.md](GOAL.md). Until then, this is a completed setup handoff, not a
running research goal.
