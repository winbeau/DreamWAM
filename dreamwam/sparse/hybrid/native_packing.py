"""Fixed-shape hard reads or fine regions plus pooled background for AV reuse.

Geometry is verified on CPU, outside graph capture but inside predict_action.
All score-dependent selection, gathering, averaging and bias construction runs
on the model device and can be captured. Pool groups never mix feature ages:
this entrypoint only accepts complete Dense anchors, enforced by configuration.
"""

import math

import torch
from torch.nn import functional as F

from ..profile.geometry import TokenGrid
from .native_scores import choose_native, native_read_route


def compile_pool_geometry(grid, full_mask_cpu, refine_fraction, device):
    if full_mask_cpu.device.type != "cpu" or full_mask_cpu.dtype != torch.bool:
        raise ValueError("validate original pool visibility on CPU before graph capture")
    nv = grid.length
    if full_mask_cpu.ndim != 2 or full_mask_cpu.shape[0] != full_mask_cpu.shape[1] or full_mask_cpu.shape[0] <= nv:
        raise ValueError("pool visibility must include original video and action positions")
    action_mask = full_mask_cpu[nv:, :nv]
    buckets = []
    packed = grid.frame_size
    regions = grid.regions()
    for frame in range(1, grid.frames):
        for size in (1, 2, 4):
            groups = [group.tolist() for group in regions if group[0] // grid.frame_size == frame and len(group) == size]
            if not groups:
                continue
            for group in groups:
                if not torch.equal(action_mask[:, group], action_mask[:, group[:1]].expand(-1, len(group))):
                    raise ValueError("pooled keys have different native AV visibility")
            refined = len(groups) if size == 1 else math.ceil(len(groups) * refine_fraction)
            packed += refined * size + len(groups) - refined
            buckets.append(dict(members=torch.tensor(groups, dtype=torch.long, device=device), refined=refined, size=size))
    return dict(frame_size=grid.frame_size, frames=grid.frames, original_length=nv,
                packed_length=packed, buckets=buckets)


def pool_read_count(grid, refine_fraction):
    """Exact geometry budget for a matched hard-selection control, without CUDA."""
    mask = torch.ones(grid.length + 1, grid.length + 1, dtype=torch.bool)
    return compile_pool_geometry(grid, mask, refine_fraction, "cpu")["packed_length"]


def pooled_layer(key, value, score, valid, full_mask, geometry, config):
    nv, frame_size = geometry["original_length"], geometry["frame_size"]
    if key.shape != value.shape or key.shape[1] != nv:
        raise ValueError("pool requires all native Dense K/V rows")
    if geometry["packed_length"] == nv:
        ids = torch.arange(nv, device=key.device)
        return dict(k=key, v=value), full_mask[nv:], ids, F.pad(ids[:, None], (0, 3), value=-1), torch.ones_like(ids)
    observed = torch.arange(frame_size, device=key.device)
    keys, values, reps = [key[:, :frame_size]], [value[:, :frame_size]], [observed]
    members = [F.pad(observed[:, None], (0, 3), value=-1)]
    sizes = [torch.ones_like(observed)]
    for bucket in geometry["buckets"]:
        groups, fine, size = bucket["members"], bucket["refined"], bucket["size"]
        count = groups.shape[0]
        group_score = None if config.signal == "uniform" else score[groups].mean(dim=1)
        chosen = choose_native(group_score, fine, count, key.device, valid=valid)
        priority = torch.arange(count, device=key.device)
        priority = priority.index_add(0, chosen, torch.full_like(chosen, count))
        background = priority.argsort(stable=True)[:count - fine]
        fine_ids = groups.index_select(0, chosen).flatten()
        coarse_ids = groups.index_select(0, background)
        keys.extend((key.index_select(1, fine_ids), key[:, coarse_ids].mean(dim=2)))
        values.extend((value.index_select(1, fine_ids), value[:, coarse_ids].mean(dim=2)))
        reps.extend((fine_ids, coarse_ids[:, 0]))
        members.extend((F.pad(fine_ids[:, None], (0, 3), value=-1), F.pad(coarse_ids, (0, 4 - size), value=-1)))
        sizes.extend((torch.ones_like(fine_ids), torch.full((count - fine,), size, dtype=torch.long, device=key.device)))
    representatives = torch.cat(reps)
    order = representatives.argsort(stable=True)
    representatives = representatives.index_select(0, order)
    group_sizes = torch.cat(sizes).index_select(0, order)
    group_members = torch.cat(members).index_select(0, order)
    k = torch.cat(keys, dim=1).index_select(1, order)
    v = torch.cat(values, dim=1).index_select(1, order)
    visibility = full_mask[nv:, :nv].index_select(1, representatives)
    bias = group_sizes.float().log() if config.multiplicity == "count" else torch.zeros_like(group_sizes, dtype=torch.float32)
    visual_bias = bias[None].expand(visibility.shape[0], -1).masked_fill(~visibility, -torch.inf)
    action_visibility = full_mask[nv:, nv:]
    action_bias = torch.zeros_like(action_visibility, dtype=torch.float32).masked_fill(~action_visibility, -torch.inf)
    mask = torch.cat((visual_bias, action_bias), dim=1)
    if k.shape[1] != geometry["packed_length"]:
        raise RuntimeError("packed region count changed with scores")
    return dict(k=k, v=v), mask, representatives, group_members, group_sizes


def pack_native_reads(config, *, kv, scores, valid, full_mask, grid_size, read_count, geometry):
    grid = TokenGrid(*grid_size)
    if config.packing == "pool" and geometry["packed_length"] != read_count:
        raise ValueError("declared read budget differs from actual fine/pool geometry")
    if config.layerwise and scores.shape[0] != len(kv):
        raise ValueError("layerwise reads need each layer's native score")
    shared_score, shared_valid = scores.mean(dim=0), valid.all()
    packed, masks, routes, members, sizes = [], [], [], [], []
    for layer, tensors in enumerate(kv):
        score = scores[layer] if config.layerwise else shared_score
        usable = valid[layer] if config.layerwise else shared_valid
        if config.packing == "pool":
            values, mask, route, group_members, group_sizes = pooled_layer(
                tensors["k"], tensors["v"], score, usable, full_mask, geometry, config)
        else:
            route = native_read_route(config, score, usable, count=read_count,
                                      frame_size=grid.frame_size, frames=grid.frames)
            values = {name: tensor.index_select(1, route) for name, tensor in tensors.items()}
            action_ids = torch.arange(full_mask.shape[0] - grid.length, device=route.device) + grid.length
            mask = full_mask[grid.length:].index_select(1, torch.cat((route, action_ids)))
            group_members = F.pad(route[:, None], (0, 3), value=-1)
            group_sizes = torch.ones_like(route)
        packed.append(values)
        masks.append(mask)
        routes.append(route)
        members.append(group_members)
        sizes.append(group_sizes)
    return dict(kv=packed, masks=masks, route=torch.stack(routes),
                members=torch.stack(members), sizes=torch.stack(sizes), fallback=~valid.all())
