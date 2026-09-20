# What the current acceleration does and does not establish

Status: **MECHANISM AUDIT; M1–M3 end-to-end claim unverified**.
Date: **2026-09-20 UTC**. This source review responds to the user's question
about action/video classification, AV–VV bridging and sparse execution. It
does not change any frozen rollout or the paper's TeX.

The current approximately **1.78×** temporal and **1.81×** guided speedups
come from visual computation reuse across denoising steps with transformer
CUDA graphs, compared with a Dense control given the same applicable common
optimizations. They are not measurements of the paper's complete M1–M3 path.
The new [interval-10 control](visual-cadence-single-factor-20260920.md) reaches
**1.973×** but has no official episode evidence and no action-guided selection.
None of these figures establishes preserved benchmark SR.

## Source-level mapping

| Paper component | Existing implementation | Used by the current fast candidates? | Evidence still required |
|---|---|---|---|
| M1: action-sensitive / visual-context-sensitive Head×Stage types and calibrated budgets | `collect_action_impact.py` supplies interventions; `SparseConfig` accepts per-head/per-stage budgets | **No.** Preserving the action expert and reducing the visual expert is branch separation, not the paper's head classification | Measured sensitivity/type profiles, their stability and matched-cost uniform/head/head×stage comparisons |
| M2: action anchors, VV dependency context and structural support | `routing.py` implements `av` / `av_context` routes and structural handling for sparse VV keys | **No** in the fast visual-cache path. The existing route's mean-Q / block-K affinity is a ranking proxy; its presence is not proof of anchor-conditioned dependency completion | Visual-only, AV-only and AV+context at equal executed budgets, with causal interventions and full paired SR/latency |
| M3: route reuse, compact/block execution and cost-aware use | Sparse attention has masked/gather implementations and route-scope reuse; earlier execution measurements were negative for net speed | The fast path uses a different approximation: reusing visual K/V and hidden outputs while computing every action step, plus CUDA graph dispatch | Attribute any additional benefit to the actual active mechanism; do not relabel cached visual computation as a successful sparse VV kernel |

The paper's classification refers to **Head×Stage decision sensitivity**, not
just the architecture's existing action/video token split. The current temporal
candidate uses a uniform refresh cadence. At full token budget it does not even
construct an action relevance signal.

The guided candidate in `action_guided_visual_token_cache.py` uses the current
first-layer action queries/keys and cached visual keys, jointly normalizes
A→[V,A], averages the resulting visual relevance and ranks token refreshes by
`input_drift * (1 + weight * mean_normalized_action_relevance)`. This is a
causal AV-conditioned drift score shared across layers. It does **not** call
the sparse VV routing path, classify heads, or construct the paper's complete
`Anchor ∪ Context ∪ Structural` route. It must not be described as having
validated those mechanisms.

## Current evidence boundaries

- The [conditioned-frame comparison](conditioned-frame-single-factor-20260920.md)
  measures 254.80 ms Dense, 142.87 ms interval-5 temporal and 140.49 ms guided.
  The 2.37 ms temporal/guided difference changes the visual refresh budget and
  adds guidance, so it is **not** an isolated action-guidance contribution.
  An earlier eager matched-budget check found guidance adds approximately
  0.40 ms; that is an overhead measurement, not a quality benefit.
- The [official coverage record](temporal-suite-expansion-20260920.md) has one
  newly completed matched Dense Spatial control, **492/500 successes (98.4%)**.
  Candidate coverage is incomplete and contains accepted task failures. Pilots,
  action-vector similarity and graph/eager equality cannot establish benchmark SR.
- New conditioned-frame and cadence timings are benchmark-only variants.
  Existing frozen official runs retain their original code/options. Do not
  attach an older configuration's SR to a newly optimized row merely because
  actions matched on the synthetic correctness inputs.
- Earlier VV attention and real-checkpoint FFN/neuron experiments failed to
  provide useful net acceleration. That evidence remains relevant; it does
  not establish that every future implementation must fail. The older 5.3%
  attention fraction applies to its measured execution path, not automatically
  to the new graph/control path.
- Current results do not prove that AV+VV is necessary, that head types are
  stable, or that action guidance preserves decisions better than drift-only
  selection. These remain experimental questions.

## Evidence needed before a paper-method claim

Complete the authorized paired SR coverage and retain the repeatability and
native-renderer limitations. For mechanism attribution, freeze one cadence,
one actual token/pair budget and one common execution backend, then compare
visual-only, AV-only and AV+VV context. Add the calibrated Head×Stage factor
in a separate measurement. Charge selection, context construction, gathers,
graph staging and CPU action output to every complete request; give Dense the
same applicable common optimizations.

This is an evidence plan, not a claim that those new experiments ran. Full
Pareto results precede the user's SR-tolerance decision. If the mechanism adds
no quality or latency benefit, report that result and narrow the paper's claim.
