# DreamWAM evaluation fork guidance

This fork maintains paper-evaluation integration on **main**, also the GitHub default branch. Preserve upstream code and attribution. FINAL STATUS: planning only; the user paused implementation again. Future priority is DreamWAM and RLinf DreamZero-5B single-GPU LIBERO evaluation; FastWAM-Joint remains shelved. Wait for an explicit request to resume a stage.

- Install and execute only on the evaluation server, never locally. Only GPU5 is authorized; do not touch other workloads.
- Preserve verified pyproject.toml and uv.lock byte-for-byte when compatible. Record source hashes and justify every necessary version change. Pin added dependencies and build tools exactly; lock on server, then use `uv sync --locked`. Never upgrade implicitly.
- The reference model environment reports torch 2.7.1+cu126, Python 3.10.20; do not silently replace it with cu128. A model lockfile has not yet been recovered/generated.
- Do not modify live environments, download weights, implement adapters or run GPU work until the user resumes execution.
- No new GitHub workflow CI. No force pushes or destructive resets. main is the normal entrypoint; upstream revisions are provenance, not permission to overwrite existing history.
- Before adapter/config work, read the corresponding skills in winbeau/action-eval (`skills/action-eval-adapter/SKILL.md`, `skills/action-eval-config/SKILL.md`). The evaluator owns success; the model only returns actions.
- README states identity, status and entrypoints; docs/action-eval indexes environment, protocol and evidence. Every verification record includes status, timestamp/timezone, commit and checkpoint hashes, command, exit code, artifacts, limitations and next step.
- Pilot is not benchmark. Errors are not task failures. Incomplete coverage must never yield a complete SR.

## Required development and deployment workflow

All projects: local clone/edit → local `git add`, `git commit`, `git push` on main → server `git pull --ff-only` → `uv sync --locked` → explicitly authorized training/evaluation. Never modify application code on the compute server or bypass Git with copied deployments. Check for server edits before pulling; never discard them. Weights/data/cache are external to Git. If a lock must be generated on the server, return it for local review/commit/push and pull the committed version before execution. This documentation task does not authorize running that workflow beyond local documentation commits.

Start with [the handoff plan](docs/action-eval/README.md).
