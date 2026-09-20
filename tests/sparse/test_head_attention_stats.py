"""The descriptive statistics must retain native joint normalization."""

import importlib.util
from pathlib import Path
import sys

import pytest
import torch


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/sparse"
sys.path.insert(0, str(SCRIPT))
spec = importlib.util.spec_from_file_location("head_attention_statistics", SCRIPT/"collect_head_attention_stats.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def uniform_case():
    # Two conditioned tokens, four future tokens, two action tokens.
    mask = torch.ones(8, 8, dtype=torch.bool)
    mask[:2, 2:] = False
    mask[2:6, 6:] = False
    return dict(video_io=(torch.zeros(1,6,4),torch.zeros(1,6,4)),
        action_io=(torch.zeros(1,2,4),torch.zeros(1,2,4)),
        attention_mask=mask,video_length=6,tokens_per_frame=2)


def test_uniform_logits_preserve_joint_av_mass_and_vv_mask():
    actual = module.attention_statistics(uniform_case(),2)
    expected = dict(av_conditioned_mass=0.25,av_future_mass=0.5,av_action_mass=0.25,
        vv_conditioned_mass=1/3,vv_future_mass=2/3,vv_action_mass=0,
        av_future_aggregate_entropy=1,vv_future_aggregate_entropy=1,
        av_vv_future_aggregate_cosine=1)
    for name,value in expected.items():
        torch.testing.assert_close(actual[name],torch.full((2,),value,dtype=torch.float32))


def test_disjoint_av_and_vv_future_preferences_lower_cosine_and_entropy():
    kwargs = uniform_case()
    kwargs["video_io"][0][:,2:,0] = 15
    kwargs["action_io"][0][:,:,1] = 15
    kwargs["video_io"][1][:,2,0] = 15
    kwargs["video_io"][1][:,3,1] = 15
    actual = module.attention_statistics(kwargs,2)
    assert actual["av_vv_future_aggregate_cosine"][0] < 1e-6
    assert actual["av_future_aggregate_entropy"][0] < 1e-6
    assert actual["vv_future_aggregate_entropy"][0] < 1e-6
    # The untouched head still has uniform aggregate distributions.
    assert actual["av_vv_future_aggregate_cosine"][1] == pytest.approx(1)


def test_reject_expanded_mask_and_batch_mismatch():
    kwargs = uniform_case()
    kwargs["attention_mask"] = kwargs["attention_mask"][None,None]
    with pytest.raises(ValueError,match="native two-dimensional"):
        module.attention_statistics(kwargs,2)
    kwargs = uniform_case()
    kwargs["video_io"] = tuple(x.expand(2,-1,-1) for x in kwargs["video_io"])
    kwargs["action_io"] = tuple(x.expand(2,-1,-1) for x in kwargs["action_io"])
    with pytest.raises(ValueError,match="batch one"):
        module.attention_statistics(kwargs,2)
