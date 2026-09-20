import pytest
import torch

from dreamwam.sparse.hybrid.profiling import DependencyAudit, balanced_indices, route_stability
from test_visual_step_cache import model_and_inputs


def test_audit_does_not_change_dense_or_leak_hooks():
    model, inputs = model_and_inputs()
    expected = model.sample_action(**inputs)
    with DependencyAudit(model, collect=True, ratio=0.25) as audit:
        assert torch.equal(model.sample_action(**inputs), expected)
    assert len(audit.records) == 4
    assert len(route_stability(audit.records)) == 9
    assert audit.future_latents.shape[2] == 2
    assert audit.modified_layers == 0
    assert torch.equal(model.sample_action(**inputs), expected)


@pytest.mark.parametrize("method", ["uniform", "action", "bottom_action", "visual_context", "action_context"])
def test_intervention_exact_scope_and_cleanup(method):
    model, inputs = model_and_inputs()
    expected = model.sample_action(**inputs)
    with DependencyAudit(model, remove_step=2, method=method, ratio=0.25) as audit:
        actual = model.sample_action(**inputs)
    assert audit.modified_layers == model.mot.num_layers
    assert len(set(audit.removed.tolist())) == 3
    assert [sum(i // 4 == frame for i in audit.removed.tolist()) for frame in range(3)] == [1, 1, 1]
    assert torch.isfinite(actual).all()
    assert not torch.equal(actual, expected)
    assert torch.equal(model.sample_action(**inputs), expected)


def test_rank_direction_and_removal_cannot_empty_frames():
    score = torch.arange(12).float()
    assert balanced_indices(score, 12, 4, 0.25).tolist() == [3, 7, 11]
    assert balanced_indices(score, 12, 4, 0.25, bottom=True).tolist() == [0, 4, 8]
    with pytest.raises(ValueError):
        balanced_indices(score, 12, 4, 0.99)
