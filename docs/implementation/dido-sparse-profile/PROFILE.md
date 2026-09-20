# Native action–video profile

Status: implemented, pending server tests and real-checkpoint capture. No new
selection, speed or SR result is asserted by this instrumentation.

`scripts/sparse/profile_action_video.py` consumes the committed experiment plan
and hash-verified self-captured observations. It refuses an unfrozen run checkout,
changed input/checkpoint, non-CUDA model fallback, or admission that would consume
the last available authorized H100. A failed/partial capture cannot be marked
complete. The script does not train or alter the evaluator.

## Measurement definitions

At each sampled layer and denoising step, capture the **actual** native projected
Q/K/V, after Q/K normalization and original RoPE. Select complete heads, never
truncate the channel width or denominator. Each raw archive stores Q/K/V,
original joint boolean mask, all A→[V,A] and V→[V,A] probabilities, grid metadata,
original projection dtype, head identities, and current input identity. Numerical
probabilities are recomputed in float32 on CPU and are descriptive; original
fused attention still produces every model output. This is offline overhead.

For each head h and visual key j:

```
P_h = softmax(Q_h K_h^T / sqrt(d) + native_mask)
d_hj = mean(action query i) P_hij
value_action_hj = d_hj * ||V_hj||_2
context_hj = sum(visual query i) [d_hi / sum_i d_hi] * P_hij
visual_context_hj = mean(visual query i) P_hij
video_time_hfj = ||V_hfj - V_h0j||_2
denoising_drift_hsj = ||V_hsj - V_h(s-1)j||_2
```

`value_action` omits signed cancellation, output projection and downstream
nonlinearity; it is our proxy, not the author's undefined action V-attribution.
`context` propagates significance backward to supporting keys. A→V denotes
action **queries** reading video **keys**. AV is not renormalized over video keys.
The action-key probability mass remains available for denominator checks.

Video-time difference and denoising drift have different axes. The former is
zero on the observed frame by construction, and neither is a real object/motion
label. Denoising drift is paired within the same request, layer, head and cell.
If sampled steps have a gap, it is recorded as a gap, never called adjacent.
Raw video latents after each sampler update and the final normalized action
sequence are saved separately; final returned actions retain original policy
denormalization/binarization. The per-step observed-frame latent is labelled
before native restoration; the final tensor is after restoration.

Camera metadata denotes nominal input-image footprints. Cell `7` is the first
wrist cell on row 0; cell `14` returns to agentview on row 1. Neither consecutive
halves of the flattened frame nor DIDO's published grid are used. Original
complex RoPE tables (or native cos/sin pairs) are archived per step. Ragged 2×2
regions partition the 7×7 camera footprints without crossing a frame or seam.

## Bounded execution and replay

Declared initial cohort: 9 observations × 10 steps × 4 layers × 4 sampled heads;
18 complete policy predictions including uninstrumented parity controls; 2 GiB
raw-array cap. Layers `[0,9,19,29]`, heads `[0,6,12,18]` are fixed before observing
scores. They are a coverage pilot, not all-layer/all-head statistics. The first
3-pair closed loop remains a separate later gate, charged to the global cap of
50 episode attempts across arms.

On an admitted H100 detached run worktree, using the existing model environment:

```bash
python scripts/sparse/profile_action_video.py \
  --inputs /root/wenbiao_zhao/dreamwam-sr/outputs/hybrid-routing-profile-20260920/trajectory-dense/manifest.json \
  --out-dir /root/wenbiao_zhao/dreamwam-sr/outputs/dido-sparse-profile-20260920/profile-REV
```

Every `raw/records.jsonl` entry gives the compressed NPZ SHA-256, byte sizes,
array dtypes/shapes, indices and semantic metadata. Load with `allow_pickle=False`.
`joint_probabilities(query,key,mask)` reconstructs the saved AV/VV arrays.
The writer refuses overwrites and stops on its byte limit; incomplete artifacts
are preserved. Runtime counters reset between requests and all hooks restore on
success or failure. These checks do not establish candidate CUDA graph parity;
that gate applies to the separate online runtime implementation.
