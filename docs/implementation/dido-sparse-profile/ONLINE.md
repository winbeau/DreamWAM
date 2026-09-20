# Native-anchor selection and background pooling

Status: implementation submitted for H100 CPU/CUDA verification. No new online
speed or SR result is claimed yet. This extends the inherited hybrid executor;
it does not replace the uniform feature-reuse control or change model inputs,
checkpoint, resolution, action horizon or denoising steps.

## Signals and their age

`selection.method: native` enables an explicit `native_routing` configuration.
The Dense executor observes its real post-RoPE Q/K and unrotated V at the actual
layer where they are used. It performs no additional projection. Up to four
evenly spaced complete heads are sampled by default (H100 `[0,6,12,18]`). Shared
routes score fixed depths `[0,9,19,29]` by default on the 30-layer model; an
explicit depth list is available as a separate ablation. Layerwise routing
scores each layer independently. All scores are computed on the model device,
inside Dense graph execution and its measured cost, without offline teacher
indices, future ground truth or annotations.

Signals follow [PROFILE.md](PROFILE.md): `action`, `value_norm`, `value_action`,
`dynamic` (video-time V difference), `visual_context`, and `action_context`
(backward VV support seeded by AV). Full original joint masks and action-key
denominators are retained. `uniform` bypasses signal arithmetic. `fusion`
requires explicit named weights; no mixture is enabled automatically. Weights
must be compared on development data before adopting a frozen candidate.

Each depth's nonnegative score is normalized to unit mean. Shared routes average
these sampled-depth vectors; layerwise routes use their own vector. This is a
predeclared aggregation hypothesis to compare with layerwise or single-depth
selection, not a learned attribution. Invalid/nonfinite score vectors fall back
to uniform indices, with a recorded event. Score computation only occurs at
Dense steps. `native_score_step` and `native_score_age` expose stale ranking
information during later sparse or reuse steps; those steps are never labelled
fresh multi-depth probes. Stable rankings do not imply that stale K/V is safe.

## Read and recompute sets

`read.keep_count` supplies an exact integer budget instead of a ratio. Counts
are checked against the actual grid; it is mutually exclusive with keep_ratio.
Legacy configurations and their fingerprints remain unchanged when new fields
are absent. Shared hard reads use the inherited set invariant at Sparse steps:
`R_new - R_old ⊆ U ⊆ R_new`. The explicit independent recompute selector is
`uniform`, current input-token `drift`, or the frozen `anchor` importance.

Observed-frame handling is explicit: `score` retains its normal rank/tie rule;
`uniform` substitutes distributed observed-frame indices; `full` retains every
observed cell and distributes the remaining budget over future frames. This
matters because dynamic observed-frame scores are identically zero. Per-frame
query budgets cannot exceed their read quota; invalid configurations are refused.

Shared hard native reads support both feature and structure modes. In structure
mode, read indices/scores from the current request's anchor are structural state;
every kept Q/K/V is freshly computed in every subsequent step. The native score
itself remains anchor-derived until the next Dense step. No cached K/V is passed
to the fresh executor.

Layerwise reads and pooling are initially restricted to **Dense/reuse feature
schedules**. Every layer may read different original positions, but there is no
hidden transfer of one layer's compact token state into another layer's indices.
A Sparse or structure combination is explicitly rejected. Global shared hard
reads retain the independent sparse-update path for the schedule ablations.

## Fine regions and background summaries

Pooling applies to AV reads from complete Dense anchors. Original visual
computations and video-path masks at those anchors remain native. The observed
frame stays fully fine. Future 2×2 regions respect the actual frame/camera grid,
including ragged edges. Regions are ranked separately within each frame and
region-size class, so the packed size is fixed before observing values:

`packed = observed_cells + Σ(size * refined_groups + remaining_groups)`

`refined_groups = ceil(refine_fraction * groups_of_that_size)`; singleton groups
remain singleton. This preserves fixed CUDA graph shapes instead of allowing
ragged top-k choices to silently change the budget. On `(3,7,14)`, fraction 0
gives 162 rows, 0.25 gives 198, and 1 gives all 294. A fair hard control uses
exactly the same packed read count and the same full observed-frame quota.

Fine groups retain all original K/V rows. Background K is averaged after its
original RoPE and V is averaged without adding a fictitious position. `count`
multiplicity adds log(group size) to the attention bias; `unit` is a separate
ablation. Original AV visibility must be identical within each group. Action
keys retain their native visibility and no group-size bias. Full refinement
restores original ordering and the original boolean mask exactly.

Groups never mix ages: packing accepts complete fresh Dense anchors, then retains
the same groups/features until the next Dense refresh. All group members have
the anchor age. Trace output includes original member IDs (padding -1), sizes,
layer-specific routes and age; counters include actual packed rows, represented
original cells and extra score work. Selection, gathers, pooling and mask/bias
construction execute through the same eager/buffered/CUDA graph dispatcher.
CPU geometry validation and all device transfers remain inside full
predict_action timing. No local-only timing or SR result can establish success.

## Remaining verification

H100 tests must cover independent raw-score replay, frame budgets, complete
partition and mask/multiplicity parity, full-budget degeneration, independent
recompute sets, request isolation, numeric fallback and actual CUDA eager/graph
consistency. Graph tests poison float, integer, boolean and complex staging
buffers between changed requests. After these gates, use a finite checkpoint
study against stronger Dense and the original uniform feature-reuse control,
then the predeclared bounded refresh scan and at most the small 3-pair pilot.
