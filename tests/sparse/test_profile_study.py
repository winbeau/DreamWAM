import copy
import json
from pathlib import Path

import numpy as np
import pytest

from dreamwam.sparse.profile.study import case_specs, error_metrics, output_diagnostics


def test_frozen_study_has_no_duplicate_cells_or_silent_budget_expansion():
    root = Path(__file__).resolve().parents[2]
    plan = json.loads((root / "docs/implementation/dido-sparse-profile/experiment-plan.json").read_text())
    design = plan["diagnostic_study"]
    cases = case_specs(design)
    assert len(cases) == design["expected_interventions"]
    assert len(cases) + len(design["input_ids"]) == design["max_predict_calls"]
    assert {case["scope"] for case in cases} == {"AV", "VV", "joint"}
    assert {case["operation"] for case in cases} == {"delete", "replace_value_zero", "recompute"}
    smaller = copy.deepcopy(design)
    smaller["max_predict_calls"] -= 1
    with pytest.raises(ValueError, match="cap exceeded"):
        case_specs(smaller)
    duplicate = copy.deepcopy(design)
    duplicate["primary_methods"].append("uniform")
    with pytest.raises(ValueError, match="duplicate"):
        case_specs(duplicate)


def test_diagnostics_separate_raw_action_execution_prefix_and_future_latents():
    baseline = dict(action=np.ones((32, 8)), raw_action=np.ones((1, 32, 8)),
                    video_latents=np.ones((1, 4, 3, 2, 2)))
    actual = {name: value.copy() for name, value in baseline.items()}
    actual["action"][20, -1] = -1
    actual["raw_action"][0, 2, 0] += 0.5
    actual["video_latents"][:, :, 0] += 10  # Current conditioning is not future prediction error.
    result = output_diagnostics(actual, baseline)
    assert result["action"]["gripper_sign_disagreements"] == 1
    assert result["executed_prefix"]["bitwise_equal"] and result["executed_prefix"]["gripper_sign_disagreements"] == 0
    assert result["raw_prefix"]["relative_l2"] > 0
    assert result["future_latents"]["bitwise_equal"]
    with pytest.raises(ValueError, match="finite"):
        error_metrics([np.nan], [0])
    with pytest.raises(ValueError, match="matching"):
        error_metrics([0, 1], [0])
