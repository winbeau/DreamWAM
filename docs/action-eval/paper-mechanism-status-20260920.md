# What the current acceleration does and does not establish

Status: **MECHANISM AUDIT; M1–M3 end-to-end claim unverified**.
Date: **2026-09-20 UTC**. This source review responds to the user's question
about action/video classification, AV–VV bridging and sparse execution. It
does not change any frozen rollout or the paper's TeX.

Update: the user subsequently requested offline classification. The
[full M1 sensitivity sweep and combined allocation test are complete](head-stage-calibration-20260920.md):
6,480 interventions, 2,160 typed units and 21 combined profile/input records.
At equal future-key budget, head-only and Head × Stage improve action proxies
over uniform allocation; Head × Stage does not beat head-only on the separate
check input's actions. Relative-type agreement is 63.5%. This supplies first
offline M1 evidence, with broader calibration, optimized execution and paired
SR still required. It does not change the active fast-policy implementations.

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
| M1: action-sensitive / visual-context-sensitive Head×Stage types and calibrated budgets | `collect_head_stage_impact.py` measures all layer/head/stage units; `compare_head_stage_allocations.py` applies calibration-only budgets in combined offline inference | **No** in the current fast candidates; **yes** in the completed offline allocation diagnostic | Broader/dose calibration and full latency/SR; the first equal-mask-budget comparison does not show incremental Stage action benefit |
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

The linked M1 replay/allocation experiments have now run; the M2/M3 and paired
SR comparisons above remain outstanding. Full Pareto results precede the
user's SR-tolerance decision. If a factor adds no quality or latency benefit,
report that result and narrow the paper's claim.
