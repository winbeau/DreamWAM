# DreamWAM paper-evaluation handoff

This page records DreamWAM Joint integration with [action-eval](https://github.com/winbeau/action-eval). **main** is the maintained evaluation branch and default. Execution has resumed under the current user goal: preserve official paired SR while targeting ≥1.5× full-request acceleration.

Current continuation: [temporal visual-cache measurement and paired pilot](visual-cache-single-factor-20260919.md), with a measured 1.655× policy-request speedup and SR still unproven. The preceding [FFN-only negative results](ffn-cache-single-factor-20260919.md) are preserved. Older pause and GPU-7-only notes below are historical; the current user allows idle or lightly occupied GPUs.

## Current evidence

Upstream: hustvl/DreamWAM, reference commit `7c35d7d094b86fc65721cd26dfbc3194addb8fd0`. Earlier server pilot: Spatial tasks 0/1/2 × 10 initial states, 30/30 success; six native aborts required recovery. This is not four-suite evidence. See action-eval `docs/verification/P3-DREAMWAM-PILOT-COMPLETE.md`.

During a briefly authorized execution phase, the original server adapter was committed on main (40a3688) and pulled on the server. It has not received the planned fixes or new correctness validation. The action-eval server reused its original lock and passed 170 platform tests; this is not model/GPU validation. The user subsequently paused implementation again. See the [authoritative handoff](https://github.com/winbeau/action-eval/blob/main/docs/plan/00-current-handoff.md).

## Future stages

1. Preserve the server dependency inventory; reference Python 3.10.20 and torch 2.7.1+cu126. Current pyproject delegates to requirements.txt; preserve both where possible. Recover/generate a complete lock only on server and record justified differences, including build-system pins.
2. Audit injected checkpoint handling, effective denoising/action-horizon options, fixed-per-predict RNG, CUDA errors and preprocessing ownership. Add contract/native parity tests before deployment.
3. Hash checkpoint, config and benchmark assets. Compare native suite-specific protocol to the evaluator; do not blindly apply Spatial horizon to all suites.
4. On explicit resumed authorization, run GPU5 minimal checks and representative four-suite timing/stability tests. Do not silently reuse earlier pilot results as formal episodes.
5. Present a launch sheet for 4 × 10 × 50 = 2,000 episodes. The ten-hour evaluation budget begins after preparation; no guarantee until measured. Prioritize this model while other baseline identities are resolved.
6. Preserve all attempts, impose finite recovery limits and report per-task/per-suite counts plus complete-coverage SR. Do not assume resume makes native aborts statistically harmless.

## Records and acceptance

Use PLANNED / VERIFIED / FAILED / BLOCKED statuses. Record timestamp/timezone, code and checkpoint hashes, environment lineage, physical GPU UUID, exact command, exit code, raw artifact location, observations and remaining limitations. Never turn an unexecuted command into a success claim.

The central [implementation checklist](https://github.com/winbeau/action-eval/blob/main/docs/plan/07-paper-baselines-checklist.md) governs the next execution session. No installation or testing commands on this page have been run as part of this documentation handoff.
