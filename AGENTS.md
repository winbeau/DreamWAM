# DreamWAM evaluation fork guidance

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

## M1/M2/M3 H200 allocation (user authorization, 2026-09-21)

The user explicitly requests that subsequent M1/M2/M3 experiments run on H200
and confirms reserving the four empty physical GPUs **0, 1, 4, 5**. This replaces
the H100 preference for this effort. GPUs 2/3 remain available to others; do not
interfere with the external CUDA/graphics work on 6/7. The user authorizes the
existing finite GPU-hold script: use separate state files per GPU, verify live
worker identity, release only this effort's holder on a card immediately before
its experiment, and respect lease expiry. Each experiment still needs fresh
admission and a spare available host GPU. Holds are not exclusive reservations.

Continue the full observation-budget/AV-VV/adaptive-routing goal in
`docs/action-eval/m1-m2-m3-goal-20260921.md`. This allocation authorizes new
experiments, not restarting historical H200 fresh-token SR or 500-episode queues.
Record the new execution/rendering cohort explicitly and keep H100 and historical
results separate. Model weights, dependency versions and benchmark protocol stay
unchanged; source updates still use local commit/push, server fast-forward pull,
and clean pinned worktrees.

## Latest allocation, 2026-09-21 12:20 UTC

The user paused H200 holding at 12:12 UTC. All four owned holders on 0/1/4/5
were identity-checked and released; do not restart them automatically. The user
then explicitly requests **H100 physical GPU 1** for holding and running the
current M1/M2/M3 work ("1卡占卡，先用1卡跑着"). GPU 1 has variable external
occupancy and is a shared placement, not an empty or exclusive card. Use bounded
real validation/experiment work, record fresh memory/utilization and sharing,
and leave GPU 0 unused by this effort. Never signal the external occupants.
Keep the verified H100 environment, checkpoint and protocol unchanged, and keep
new H100 shared-load evidence separate from the H200 capture and timing cohort.
