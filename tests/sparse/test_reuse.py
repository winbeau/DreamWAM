"""Tests for the Context/Action Cache selection rules.

These decide *what gets reused*, so a wrong ranking silently degrades the model while looking
like a valid optimisation. Every rule is checked against a hand-computed expectation rather
than against another implementation of itself.
"""

from __future__ import annotations

import pytest
import torch

from dreamwam.sparse.reuse import (
    action_guided_score,
    down_weight_norm,
    group_masks,
    merge_reused_output,
    neuron_contribution,
    select_neurons,
    select_recompute_tokens,
    token_drift,
)


# --- Context Cache: token drift ---------------------------------------------------


def test_drift_is_zero_for_identical_inputs():
    inputs = torch.randn(1, 6, 4)
    assert torch.allclose(token_drift(inputs, inputs.clone()), torch.zeros(1, 6))


def test_drift_is_one_when_a_token_moves_by_its_own_magnitude():
    """RMS(x - x_ref) / RMS(x) is 1 when the change equals the current magnitude."""
    current = torch.zeros(1, 1, 8)
    current[0, 0, :] = 1.0
    reference = torch.zeros(1, 1, 8)
    assert token_drift(current, reference).item() == pytest.approx(1.0)


def test_drift_ranks_the_token_that_actually_moved():
    current = torch.zeros(1, 4, 8)
    reference = torch.zeros(1, 4, 8)
    current[0, 2, :] = 5.0  # only token 2 moved
    drift = token_drift(current, reference)
    selected = select_recompute_tokens(drift, 0.25)
    assert selected.flatten().tolist() == [2]


def test_drift_survives_a_zero_reference_token():
    """A reference token that was all zeros must not divide by zero."""
    current = torch.zeros(1, 2, 4)
    current[0, 1, :] = 1.0
    reference = torch.zeros(1, 2, 4)
    drift = token_drift(current, reference)
    assert torch.isfinite(drift).all()


def test_recompute_selection_size_follows_the_ratio():
    drift = torch.rand(1, 10)
    for ratio, expected in ((0.0, 1), (0.3, 3), (0.5, 5), (1.0, 10)):
        assert select_recompute_tokens(drift, ratio).shape[-1] == expected


def test_recompute_ratio_is_validated():
    with pytest.raises(ValueError, match="ratio"):
        select_recompute_tokens(torch.rand(1, 4), 1.5)


def test_selection_is_per_row_not_global():
    """Each row picks its own tokens; a shared mask would have to be reduced first."""
    drift = torch.tensor([[0.9, 0.1, 0.0], [0.0, 0.2, 0.8]])
    selected = select_recompute_tokens(drift, 1 / 3)
    assert selected[0].tolist() == [0]
    assert selected[1].tolist() == [2]


# --- Action Cache: neuron contribution --------------------------------------------


def test_neuron_contribution_matches_the_hand_computed_product():
    activations = torch.tensor([[[1.0, -2.0, 0.5]]])  # [1, 1 token, 3 neurons]
    weight_norm = torch.tensor([2.0, 1.0, 4.0])
    score = neuron_contribution(activations, weight_norm=weight_norm)
    expected = torch.tensor([[1.0 * 2.0, 2.0 * 1.0, 0.5 * 4.0]])
    assert torch.allclose(score, expected)


def test_neuron_contribution_sums_over_tokens():
    activations = torch.tensor([[[1.0], [1.0], [1.0]]])  # 3 tokens, 1 neuron
    score = neuron_contribution(activations, weight_norm=torch.tensor([3.0]))
    assert score.item() == pytest.approx(9.0)


def test_down_weight_norm_is_the_row_l2():
    weight = torch.tensor([[3.0, 4.0], [0.0, 1.0]])
    assert torch.allclose(down_weight_norm(weight), torch.tensor([5.0, 1.0]))


def test_neuron_selection_keeps_the_highest_contributors():
    contribution = torch.tensor([[0.1, 9.0, 0.2, 8.5]])
    selected = select_neurons(contribution, 0.5)
    assert set(selected.flatten().tolist()) == {1, 3}


def test_neuron_width_mismatch_is_rejected():
    with pytest.raises(ValueError, match="disagree"):
        neuron_contribution(torch.randn(1, 2, 4), weight_norm=torch.rand(5))


# --- mask sharing across denoising steps -------------------------------------------


def test_group_mask_covers_every_step_in_its_group():
    """A neuron that matters in any step of a group must survive the shared mask."""
    steps, neurons = 4, 6
    contribution = torch.zeros(steps, neurons)
    contribution[0, 0] = 10.0  # matters only in the first step of group 0
    contribution[3, 5] = 9.0  # matters only in the last step of group 1
    masks = group_masks(contribution, denoising_steps=steps, ratio=2 / neurons, group_size=2)
    assert masks.shape == (2, 2)
    assert 0 in masks[0].flatten().tolist(), "step 0's neuron must survive group 0"
    assert 5 in masks[1].flatten().tolist(), "step 3's neuron must survive group 1"


def test_group_mask_uses_one_mask_per_group():
    contribution = torch.rand(6, 8)
    masks = group_masks(contribution, denoising_steps=6, ratio=0.5, group_size=3)
    assert masks.shape == (2, 4)
    # Every step inside a group shares the identical mask by construction.
    assert masks.shape[0] == 2


def test_group_mask_pads_a_partial_group():
    contribution = torch.rand(5, 4)
    masks = group_masks(contribution, denoising_steps=5, ratio=0.5, group_size=2)
    assert masks.shape == (3, 2)  # ceil(5/2) groups


def test_group_mask_validates_its_arguments():
    contribution = torch.rand(4, 4)
    with pytest.raises(ValueError, match="denoising_steps"):
        group_masks(contribution, denoising_steps=0, ratio=0.5, group_size=2)
    with pytest.raises(ValueError, match="group_size"):
        group_masks(contribution, denoising_steps=4, ratio=0.5, group_size=0)
    with pytest.raises(ValueError, match="one entry per denoising step"):
        group_masks(contribution, denoising_steps=7, ratio=0.5, group_size=2)


# --- the paper's extension: action-guided selection --------------------------------


def test_zero_weight_reproduces_the_pure_similarity_ranking():
    score = torch.tensor([[0.2, 0.8, 0.5]])
    relevance = torch.tensor([[9.0, 0.0, 1.0]])
    blended = action_guided_score(score, relevance, weight=0.0)
    assert torch.equal(blended.argsort(descending=True), score.argsort(descending=True))


def test_action_relevance_can_change_the_ranking():
    """A token the action depends on must be able to outrank a more similar one."""
    similarity = torch.tensor([[0.9, 0.5, 0.1]])
    relevance = torch.tensor([[0.0, 1.0, 0.0]])
    pure = action_guided_score(similarity, relevance, weight=0.0)
    guided = action_guided_score(similarity, relevance, weight=1.0)
    assert int(pure.argmax()) == 0, "the most similar token wins on similarity alone"
    assert int(guided.argmax()) == 1, "the action-relevant token must be able to overtake"


def test_action_guided_score_is_scale_invariant():
    """Rescaling a signal must not change the blend, so `weight` keeps its meaning."""
    relevance = torch.tensor([[0.0, 1.0, 0.4]])
    base = action_guided_score(torch.tensor([[0.9, 0.5, 0.1]]), relevance, weight=1.0)
    scaled = action_guided_score(
        torch.tensor([[900.0, 500.0, 100.0]]), relevance, weight=1.0
    )
    assert torch.allclose(base, scaled)


def test_blend_saturates_when_a_signal_is_binary():
    """Documented limitation: with two items min-max maps both signals to {0, 1}."""
    blended = action_guided_score(
        torch.tensor([[0.9, 0.1]]), torch.tensor([[0.0, 1.0]]), weight=1.0
    )
    assert torch.allclose(blended, torch.ones(1, 2)), "equal weight ties on two items"


def test_action_guided_score_validates_its_arguments():
    with pytest.raises(ValueError, match="must match"):
        action_guided_score(torch.rand(1, 3), torch.rand(1, 4), weight=1.0)
    with pytest.raises(ValueError, match="non-negative"):
        action_guided_score(torch.rand(1, 3), torch.rand(1, 3), weight=-1.0)


# --- reconstruction ---------------------------------------------------------------


def test_merge_replaces_only_the_recomputed_rows():
    reference = torch.zeros(1, 5, 2)
    recomputed = torch.ones(1, 2, 2) * 7.0
    index = torch.tensor([[1, 3]])
    merged = merge_reused_output(reference, recomputed, index)
    assert torch.allclose(merged[0, 1], torch.full((2,), 7.0))
    assert torch.allclose(merged[0, 3], torch.full((2,), 7.0))
    for row in (0, 2, 4):
        assert torch.allclose(merged[0, row], torch.zeros(2))


def test_merge_does_not_mutate_the_reference():
    reference = torch.zeros(1, 3, 2)
    merge_reused_output(reference, torch.ones(1, 1, 2), torch.tensor([[0]]))
    assert torch.allclose(reference, torch.zeros(1, 3, 2)), "reference must be left intact"


def test_merge_validates_shapes():
    with pytest.raises(ValueError, match="row count"):
        merge_reused_output(torch.zeros(1, 4, 2), torch.ones(1, 2, 2), torch.tensor([[0]]))
    with pytest.raises(ValueError, match="feature width"):
        merge_reused_output(torch.zeros(1, 4, 3), torch.ones(1, 1, 2), torch.tensor([[0]]))
