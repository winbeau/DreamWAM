"""Reuse-selection primitives for the Context Cache and the Action Cache.

These implement the two scoring rules a muscle-memory style cache needs, plus the extension
this project's method line requires: making the selection *action-guided* rather than purely
similarity-guided.

* :func:`token_drift` / :func:`select_recompute_tokens` - Context Cache. Rank visual tokens by
  how far their FFN input moved from a reference execution, and recompute only the top
  fraction. Scoring uses inputs only, so deciding what to reuse never requires evaluating the
  dense FFN first.
* :func:`neuron_contribution` / :func:`select_neurons` - Action Cache. Rank hidden neurons by
  their contribution to the FFN output and keep the top fraction, so only their down-projection
  weights need to be loaded.
* :func:`group_masks` - share one neuron mask across consecutive denoising steps, trading a
  little step-specific coverage for far fewer weight gathers.
* :func:`action_guided_score` - blend an action-relevance signal (the A->V anchor mass this
  project already computes) into either ranking, which is how the paper's decision-preserving
  criterion enters a similarity-based cache.

Everything here is pure tensor arithmetic with no model dependency, so the selection rules are
tested without a checkpoint or a GPU.  Nothing in this module decides *whether* the reuse is
numerically acceptable; that is an end-to-end quality question owned by the closed-loop SR
evaluation.
"""

from __future__ import annotations

import torch


def token_drift(
    current_input: torch.Tensor,
    reference_input: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Per-token relative input drift, ``[..., tokens]``.

    ``RMS(x - x_ref) / max(RMS(x), eps)`` over the feature axis: a token whose input barely
    moved scores near zero, and a token whose input moved by its own magnitude scores near
    one.  Both arguments are ``[..., tokens, features]``.
    """
    if current_input.shape != reference_input.shape:
        raise ValueError(
            "current and reference inputs must match: "
            f"{tuple(current_input.shape)} vs {tuple(reference_input.shape)}"
        )
    difference = current_input - reference_input
    drift = difference.pow(2).mean(dim=-1).sqrt()
    magnitude = current_input.pow(2).mean(dim=-1).sqrt().clamp(min=eps)
    return drift / magnitude


def select_recompute_tokens(drift: torch.Tensor, ratio: float) -> torch.Tensor:
    """Indices of the tokens to recompute, ``[..., keep]``, highest drift first.

    Selection is per trailing axis so each row (batch, head, layer, ...) can pick its own
    tokens; callers that want one shared selection across rows should reduce ``drift`` first.
    """
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"ratio must be in [0, 1], got {ratio}")
    count = int(round(ratio * drift.shape[-1]))
    count = max(1, min(count, drift.shape[-1]))
    return drift.argsort(dim=-1, descending=True, stable=True)[..., :count]


def neuron_contribution(
    post_activation: torch.Tensor,
    *,
    weight_norm: torch.Tensor,
) -> torch.Tensor:
    """Contribution score per hidden neuron, ``[..., neurons]``.

    ``(sum_r |h_r,n|) * ||W_n||_2``: how much a neuron's activation can move the FFN output,
    combining the activation magnitude actually produced with the norm of the row it writes
    through.  ``post_activation`` is ``[..., tokens, neurons]``.

    ``weight_norm`` is ``[neurons]`` and is required rather than derived, because deriving it
    means reading the whole down-projection on every call - which is exactly the weight traffic
    the Action Cache exists to avoid.  Use :func:`down_weight_norm` once at setup.
    """
    if post_activation.shape[-1] != weight_norm.shape[-1]:
        raise ValueError(
            "activation width and weight norm disagree: "
            f"{post_activation.shape[-1]} vs {weight_norm.shape[-1]}"
        )
    activation_mass = post_activation.abs().sum(dim=-2)
    return activation_mass * weight_norm.to(post_activation.dtype)


def down_weight_norm(down_weight: torch.Tensor) -> torch.Tensor:
    """``||W_n||_2`` per output row, ``[neurons]``; compute once, reuse for every call."""
    if down_weight.ndim != 2:
        raise ValueError(f"down_weight must be [neurons, features], got {down_weight.ndim}D")
    return down_weight.float().pow(2).sum(dim=-1).sqrt()


def select_neurons(contribution: torch.Tensor, ratio: float) -> torch.Tensor:
    """Indices of the neurons to evaluate, ``[..., keep]``, highest contribution first."""
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"ratio must be in [0, 1], got {ratio}")
    count = int(round(ratio * contribution.shape[-1]))
    count = max(1, min(count, contribution.shape[-1]))
    return contribution.argsort(dim=-1, descending=True, stable=True)[..., :count]


def group_masks(
    contribution: torch.Tensor,
    *,
    denoising_steps: int,
    ratio: float,
    group_size: int,
) -> torch.Tensor:
    """Neuron indices shared by each group of consecutive denoising steps.

    Returns ``[..., num_groups, keep]``.  A mask per step would force a weight gather per
    step; grouping amortizes that over ``group_size`` steps at the cost of some step-specific
    coverage.  The group's mask is chosen from the summed contribution over its steps, so a
    neuron that matters in any step of the group survives.
    """
    if denoising_steps <= 0:
        raise ValueError(f"denoising_steps must be positive, got {denoising_steps}")
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    if contribution.shape[-2] != denoising_steps:
        raise ValueError(
            "contribution must have one entry per denoising step on the second-to-last axis, "
            f"got {contribution.shape[-2]} for {denoising_steps} steps"
        )
    num_groups = (denoising_steps + group_size - 1) // group_size
    padded = contribution
    if num_groups * group_size != denoising_steps:
        pad = num_groups * group_size - denoising_steps
        filler = contribution[..., -1:, :].expand(*contribution.shape[:-2], pad, contribution.shape[-1])
        padded = torch.cat([contribution, filler], dim=-2)
    grouped = padded.reshape(*contribution.shape[:-2], num_groups, group_size, contribution.shape[-1])
    return select_neurons(grouped.sum(dim=-2), ratio)


def action_guided_score(
    similarity_score: torch.Tensor,
    action_relevance: torch.Tensor,
    *,
    weight: float,
) -> torch.Tensor:
    """Blend an action-relevance signal into a similarity ranking.

    Both inputs are ``[..., tokens]`` (or ``[..., neurons]``); each is min-max normalised along
    the last axis so ``weight`` has a comparable meaning regardless of their scales.  With
    ``weight = 0`` the ranking is the pure similarity ranking, which is the ablation that
    separates "reuse what looks alike" from "reuse what the action depends on".
    """
    if similarity_score.shape != action_relevance.shape:
        raise ValueError(
            "score and relevance must match: "
            f"{tuple(similarity_score.shape)} vs {tuple(action_relevance.shape)}"
        )
    if weight < 0.0:
        raise ValueError(f"weight must be non-negative, got {weight}")

    def normalize(values: torch.Tensor) -> torch.Tensor:
        low = values.amin(dim=-1, keepdim=True)
        high = values.amax(dim=-1, keepdim=True)
        return (values - low) / (high - low).clamp(min=1e-6)

    return normalize(similarity_score) + weight * normalize(action_relevance)


def merge_reused_output(
    reference_output: torch.Tensor,
    recomputed: torch.Tensor,
    recompute_index: torch.Tensor,
) -> torch.Tensor:
    """Scatter recomputed rows onto a copy of the reference output.

    This is the Context Cache's reconstruction: the caller evaluates only
    ``recompute_index`` rows and every other row keeps the reference value.
    ``reference_output`` is ``[..., tokens, features]``, ``recomputed`` is
    ``[..., keep, features]`` and ``recompute_index`` is ``[..., keep]``.

    Instruction and robot-state rows must be part of the recompute set; that is a policy
    decision made where the rows are known, not here.
    """
    if recomputed.shape[:-2] != recompute_index.shape[:-1]:
        raise ValueError(
            "recomputed rows and index must agree on leading axes: "
            f"{tuple(recomputed.shape)} vs {tuple(recompute_index.shape)}"
        )
    if recomputed.shape[:-1] != recompute_index.shape:
        raise ValueError(
            "recomputed and index must agree on the row count: "
            f"{recomputed.shape[-2]} vs {recompute_index.shape[-1]}"
        )
    if recomputed.shape[-1] != reference_output.shape[-1]:
        raise ValueError(
            "feature width must match the reference: "
            f"{recomputed.shape[-1]} vs {reference_output.shape[-1]}"
        )
    merged = reference_output.clone()
    index = recompute_index.unsqueeze(-1).expand_as(recomputed)
    return merged.scatter(-2, index, recomputed)
