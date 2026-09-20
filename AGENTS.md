# DreamWAM evaluation fork guidance

## Active DIDO goal amendments (2026-09-20)

The user invoked the full `docs/implementation/dido-sparse-profile/GOAL.md`
in this session. The setup-only hold is over for this branch. The user has no
private DIDO raw-data link; proceed with explicitly labelled DreamWAM observations
and Q/K/V, retaining the missing author-data limitation. Maximum permitted SR
drop is **5 percentage points** against matched Dense. Keep closed-loop tests
small: this effort adopts **50 episodes total across Dense and all candidate
arms, including attempted episodes**, not 50 pairs or 50 per arm. Start with a
predeclared 3-pair development pilot only after offline gates. No automatic
expansion. This supersedes the historical unspecified tolerance and 50-per-arm
screening scope below. See `docs/implementation/dido-sparse-profile/PROGRESS.md`.

## Active isolated-worktree handoff (2026-09-20)

The user explicitly requested a separate DIDO-guided action/video sparse-profile
worktree. In this checkout, work on **experiment/dido-sparse-profile**, not main.
Local root: `/home/winbeau/Papers/ICLR2027-WAM-SA/.trees/dido-sparse-profile`.
H100 root: `/root/wenbiao_zhao/dreamwam-sr/.trees/dido-sparse-profile`.
This branch-specific instruction supersedes the main-only workflow below; all
environment, protocol, provenance and shared-GPU safety rules still apply.

Read `docs/implementation/dido-sparse-profile/WORKFLOW.md` for the verified update
chain and `docs/implementation/dido-sparse-profile/GOAL.md` for the future task.
The setup turn is **handoff only**: no new profiling, training, rollout or GPU job
is started until the user invokes the goal in the new session. Do not confuse a
saved goal prompt with an already active goal. Do not spawn sub-agents without
explicit user authorization.

Edit locally, commit and push this branch, then use
`bash deployment/h100/sync-dido-worktree.sh` to fast-forward the clean H100
worktree and check exact source equality. That script never installs dependencies
or launches inference. Run tests and experiments only on H100, with immutable
per-run revisions. Do not pull into a checkout while a live run uses it; use
separate detached run worktrees. Preserve the existing main and old run worktrees.

This fork maintains paper-evaluation integration on **main**, also the GitHub default branch. Preserve upstream code and attribution. CURRENT STATUS: the user resumed Sparse-WAM on 2026-09-20 and authorized GPUs **4–7**, asking for **10% visual tokens recomputed at every denoising step, without cross-step visual caches**. This supersedes the 06:21 UTC pause for this new experiment, but does not resume the old 500-episode queue. The earlier four-suite LIBERO baseline run has finished for Spatial/Object/Goal and is incomplete for Long.

- Install and execute only on the evaluation server, never locally. The current user goal (renewed 2026-09-19) permits idle or lightly occupied GPUs for SR-constrained Sparse-WAM work, superseding the earlier GPU-7-only restriction. Recheck utilization and memory before each launch, declare sharing, and never signal another user's process. Follow action-eval's current shared-host rule to leave the last available card unused.
- Preserve verified pyproject.toml and uv.lock byte-for-byte when compatible. Record source hashes and justify every necessary version change. Pin added dependencies and build tools exactly; lock on server, then use `uv sync --locked`. Never upgrade implicitly.
- The reference model environment reports torch 2.7.1+cu126, Python 3.10.20; do not silently replace it with cu128. A model lockfile has not yet been recovered/generated.
- The adapter at [`evaluation/action_eval/infer.py`](evaluation/action_eval/infer.py) loads the checkpoint the platform injects, applies option overrides to the policy's own evaluation dict, and refuses to fall back to CPU. Re-read it before changing model wiring.
- No new GitHub workflow CI. No force pushes or destructive resets. main is the normal entrypoint; upstream revisions are provenance, not permission to overwrite existing history.
- Before adapter/config work, read the corresponding skills in winbeau/action-eval (`skills/action-eval-adapter/SKILL.md`, `skills/action-eval-config/SKILL.md`). The evaluator owns success; the model only returns actions.
- README states identity, status and entrypoints; docs/action-eval indexes environment, protocol and evidence. Every verification record includes status, timestamp/timezone, commit and checkpoint hashes, command, exit code, artifacts, limitations and next step.
- Pilot is not benchmark. Errors are not task failures. Incomplete coverage must never yield a complete SR.
- Current Sparse-WAM screening scale (user amendment, 2026-09-20): use 50 episodes per candidate, covering all 10 Spatial tasks and five initial states each. Stop this effort's 500-episode queue and legacy recovery jobs, preserve their accepted outcomes and original manifests, and defer unified full-suite tests until the method is selected. Label the new matched Dense/Sparse runs exploratory; do not apply this scope change to another effort's baseline jobs.
- Fresh-token status at 2026-09-20 11:31 UTC: the user chose “先保留结果，等空卡再测 SR”. Preserve the archived results and defer the new 50-pair SR comparison until an empty renderer is available on authorized GPUs 4–7; recheck admission and native rendering before rollout. Model `6c52f36` passes CPU/CUDA and real-checkpoint adapter checks; 96 shared-load timings yield 1.191× against stronger Dense, with large action-vector differences. The GPU-4-policy / GPU-5-shared-graphics smoke (`599d225` launcher, evaluator `e0d9e80`) failed on its first native RGB read, exit 73, with zero accepted outcomes. All owned processes exited. Do not retry that unchanged shared-renderer placement or infer SR from the error. The existing environment has no loader-discoverable OSMesa library and has not been modified.

## Required development and deployment workflow

All projects: local clone/edit → local `git add`, `git commit`, `git push` on main → server `git pull --ff-only` → verified environment → authorized evaluation. Never modify application code on the compute server or bypass Git with copied deployments. Check for server edits before pulling; never discard them. Pin live runs to separate Git worktrees. Weights/data/cache are external to Git. When a compatible committed lock exists, use `uv sync --locked`; the current DreamWAM environment has no recovered lock and is reused unchanged, not implicitly synced or upgraded. If a lock must be generated on the server, return it for local review/commit/push and pull the committed version before execution. The current acceleration goal authorizes implementation and evaluation; it does not authorize training or changing the scientific protocol.

Start with [the handoff plan](docs/action-eval/README.md).

## H100 deployment (user authorization, 2026-09-20)

The user additionally authorizes deployment and experiments via `ssh h100-server`
under the existing `/root/wenbiao_zhao` directory. Use its separate `dreamwam-sr`
subdirectory, frozen model `6c52f36` and evaluator `e0d9e80`; do not restart the
deferred H200 SR runs. The initial H100 inventory has six visible devices: use
policy GPU 3 / renderer GPU 4 with GPU 5 unused, subject to fresh admission.
Pass `--authorized-gpus 3 4 5` to the bounded launcher. The user explicitly
requires Hugging Face downloads through `hf-mirror.com`; verify against the
reference H200 hashes. Preserve model and simulator versions. Keep required
graphics libraries private to the deployment and matched to driver 590.48.01.
H100 results form a new hardware/rendering cohort, with 50 episodes per arm;
do not pool them with H200 timing or historical outcomes.
