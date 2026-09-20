"""Matched-cost allocation and no leakage from the check-input metrics."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch


SCRIPT=Path(__file__).resolve().parents[2]/"scripts/sparse"
sys.path.insert(0,str(SCRIPT))
spec=importlib.util.spec_from_file_location("head_stage_allocations",SCRIPT/"compare_head_stage_allocations.py")
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def native_mask():
    mask=torch.ones(8,8,dtype=torch.bool)
    mask[:2,2:]=False
    mask[2:6,6:]=False
    return mask


def test_allocations_have_equal_executed_pairs_and_preserve_structural_rows():
    native=native_mask()
    make=lambda budgets:module.allocated_mask(native,budgets,video_length=6,tokens_per_frame=2,block_size=2)
    uniform,heterogeneous=make([1]*4),make([0,1,2,1])
    assert uniform[:,:,:6,:6].sum()==heterogeneous[:,:,:6,:6].sum()
    for mask in [uniform,heterogeneous]:
        assert torch.equal(mask[:,:,6:],native[6:][None,None].expand(1,4,-1,-1))
        assert torch.equal(mask[:,:,:2],native[:2][None,None].expand(1,4,-1,-1))
        assert mask[:,:,2:6,:2].all()
    assert not heterogeneous[0,0,2:6,2:6].any()
    assert heterogeneous[0,2,2:6,2:6].all()
    assert torch.equal(make([2]*4),native[None,None].expand(1,4,-1,-1))


def test_action_rows_remain_bitwise_under_real_attention():
    generator=torch.Generator().manual_seed(7)
    q,k,v=[torch.randn(1,4,8,8,generator=generator) for _ in range(3)]
    native=native_mask()[None,None].expand(1,4,-1,-1).clone()
    changed=module.allocated_mask(native[0,0],[0,1,2,1],video_length=6,tokens_per_frame=2,block_size=2)
    dense=torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=native)
    sparse=torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=changed)
    assert torch.equal(dense[:,:,6:],sparse[:,:,6:])
    assert torch.equal(dense[:,:,0:2],sparse[:,:,0:2])
    assert not torch.equal(dense[:,:,2:6],sparse[:,:,2:6])


def test_profiles_rank_calibration_only_and_switch_by_stage_at_fixed_cost():
    entries=[dict(layer=0,head=h,stage=s,calibration_action_relative_l2=float(23-h if s==1 else h),
                  check_action_relative_l2=float(h*1000)) for h in range(24) for s in range(3)]
    profiles=module.allocations(entries,layers=1)
    for name,profile in profiles.items():
        if name!='dense':assert np.all(np.asarray(profile).sum(axis=-1)==168)
    assert np.flatnonzero(profiles['head_only'][0][1]).tolist()==list(range(12,24))
    assert np.flatnonzero(profiles['head_stage'][0][1]).tolist()==list(range(12))
    for entry in entries:entry['check_action_relative_l2']=-1e20
    assert profiles==module.allocations(entries,layers=1)
    with pytest.raises(ValueError,match='exactly once'):
        module.allocations(entries+[entries[0]],layers=1)


def test_reject_fractional_or_out_of_range_budgets():
    for budgets in [[0.5],[-1],[3],[True],[]]:
        with pytest.raises(ValueError,match='integer'):
            module.allocated_mask(native_mask(),budgets,video_length=6,tokens_per_frame=2,block_size=2)
