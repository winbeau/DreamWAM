# DreamWAM evaluation fork guidance

This fork maintains paper-evaluation integration on **main**, also the GitHub default branch. Preserve upstream code and attribution. CURRENT STATUS: the Sparse-WAM line (branch work on main) is active; the earlier four-suite LIBERO baseline run has finished for Spatial/Object/Goal and is incomplete for Long.

- Install and execute only on the evaluation server, never locally. Physical GPU **7** is the only authorized card for Sparse-WAM work as of 2026-09-19; GPU 4 was used for two early read-only S1 measurements and must not be used again, and GPUs 0-6 plus every other user's process are off limits. GPU 7 may carry small foreign processes: coexist with them, never signal or terminate them. Re-confirm the window before each new run.
- Preserve verified pyproject.toml and uv.lock byte-for-byte when compatible. Record source hashes and justify every necessary version change. Pin added dependencies and build tools exactly; lock on server, then use `uv sync --locked`. Never upgrade implicitly.
- The reference model environment reports torch 2.7.1+cu126, Python 3.10.20; do not silently replace it with cu128. A model lockfile has not yet been recovered/generated.
- The adapter at [`evaluation/action_eval/infer.py`](evaluation/action_eval/infer.py) loads the checkpoint the platform injects, applies option overrides to the policy's own evaluation dict, and refuses to fall back to CPU. Re-read it before changing model wiring.
- No new GitHub workflow CI. No force pushes or destructive resets. main is the normal entrypoint; upstream revisions are provenance, not permission to overwrite existing history.
- Before adapter/config work, read the corresponding skills in winbeau/action-eval (`skills/action-eval-adapter/SKILL.md`, `skills/action-eval-config/SKILL.md`). The evaluator owns success; the model only returns actions.
- README states identity, status and entrypoints; docs/action-eval indexes environment, protocol and evidence. Every verification record includes status, timestamp/timezone, commit and checkpoint hashes, command, exit code, artifacts, limitations and next step.
- Pilot is not benchmark. Errors are not task failures. Incomplete coverage must never yield a complete SR.

## Required development and deployment workflow

All projects: local clone/edit → local `git add`, `git commit`, `git push` on main → server `git pull --ff-only` → `uv sync --locked` → explicitly authorized training/evaluation. Never modify application code on the compute server or bypass Git with copied deployments. Check for server edits before pulling; never discard them. Weights/data/cache are external to Git. If a lock must be generated on the server, return it for local review/commit/push and pull the committed version before execution. This documentation task does not authorize running that workflow beyond local documentation commits.

Start with [the handoff plan](docs/action-eval/README.md).
