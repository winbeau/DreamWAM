# DreamWAM paper-evaluation handoff

This page records DreamWAM Joint integration with [action-eval](https://github.com/winbeau/action-eval). **main** is the maintained evaluation branch and default. Execution has resumed under the current user goal: preserve official paired SR while targeting ≥1.5× full-request acceleration.

Current continuation: the [conservative temporal graph path and stronger native Dense control](temporal-graph-single-factor-20260919.md) measure **149.60 ms versus 293.96 ms (1.965×)** for full requests. Guided 10% takes **143.70 ms (2.046×)** against that same control. All 360 outputs are bitwise their own eager path, following the [separate transformer graph factors](visual-graph-dispatch-single-factor-20260919.md). The conservative graph pilot completed 15/15 paired successes and an independent 500-identity run has started. At 2026-09-20 00:12 UTC, its coverage was 6/500; matched Dense had 274/500 terminal outcomes (272 successes, 2 task failures). Eager temporal retains 142/500 successes, inactive after a GPU resource gate. The [eager guided candidate](visual-token-cache-single-factor-20260919.md) retains 95/500 terminal outcomes including two failures where Dense and temporal succeeded, and remains inactive. The small pilots do not establish quality preservation; incomplete full SR is withheld. Eager temporal timing remains 1.655× for two visual refreshes and 1.803× for one. [FFN/neuron negative results](ffn-cache-single-factor-20260919.md) are preserved. The user permits idle or lightly occupied GPUs; historical pause and GPU-7-only notes no longer govern this work.

The first-input audit also found different camera-image hashes before any policy-dependent action, including 10 same-Dense identities on the same GPU with identical state vectors. Hashes do not quantify image error, but repeatable reference observations have not been established. The successful pilot counts therefore do not isolate a decision-preservation effect or prove non-inferiority. See the [input audit and current recovery state](visual-token-cache-single-factor-20260919.md#first-input-repeatability-is-not-established).

## Provenance and verification

Upstream: hustvl/DreamWAM, reference commit `7c35d7d094b86fc65721cd26dfbc3194addb8fd0`. Earlier server pilot: Spatial tasks 0/1/2 × 10 initial states, 30/30 success; six native aborts required recovery. This is not four-suite evidence. See action-eval `docs/verification/P3-DREAMWAM-PILOT-COMPLETE.md`.

The adapter loads the injected checkpoint, reports executed options, preserves preprocessing ownership and refuses silent CPU fallback. Graph policy integration at **2c02c5c** passed **206 sparse tests** on the server, including actual CUDA graph execution through the adapter. Native-equivalent matched controls pass bitwise action comparisons. These checks establish implementation behavior; only complete official episode pairing can establish measured benchmark quality. The separate baseline-suite effort is tracked in the [action-eval handoff](https://github.com/winbeau/action-eval/blob/main/docs/plan/00-current-handoff.md).

## Remaining work

1. Preserve the server dependency inventory; reference Python 3.10.20 and torch 2.7.1+cu126. Current pyproject delegates to requirements.txt; preserve both where possible. Recover/generate a complete lock only on server and record justified differences, including build-system pins.
2. Complete all 500 Spatial identities for each frozen candidate and the matched Dense control. Pilot records remain separate and cannot fill a full-run denominator.
3. Evaluate action guidance with its own matched budget/cadence control. The current 1.657× latency measurement does not establish a quality benefit from guidance.
4. Preserve all attempts, impose finite recovery limits and report complete coverage before SR. Native EGL aborts remain unresolved; do not assume resume makes them statistically harmless or apply unverified rendering workarounds.
5. Keep further optimizations separate, with complete `predict_action` timing and their own quality evidence. Any extension to another suite must use its official protocol; do not reuse Spatial's horizon blindly.

## Records and acceptance

Use PLANNED / VERIFIED / FAILED / BLOCKED statuses. Record timestamp/timezone, code and checkpoint hashes, environment lineage, physical GPU UUID, exact command, exit code, raw artifact location, observations and remaining limitations. Never turn an unexecuted command into a success claim.

The central [implementation checklist](https://github.com/winbeau/action-eval/blob/main/docs/plan/07-paper-baselines-checklist.md) tracks baseline-suite integration. The linked factor records above contain the commands and artifacts actually executed for this acceleration goal. No dependency installation was performed during these factor measurements.
