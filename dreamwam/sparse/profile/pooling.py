"""Reference K/V region pooling with explicit positions, masks and multiplicity.

Numerical diagnostic only. The online packed executor must account for its own
scoring/packing costs; reference pooling does not establish an inference speedup.
"""

from dataclasses import dataclass
import math

import torch

from .geometry import TokenGrid


@dataclass(frozen=True)
class PoolPlan:
    groups: tuple[tuple[int, ...], ...]
    grid: TokenGrid
    multiplicity: str

    @classmethod
    def from_grid(cls, grid, *, refined_regions, pool_observed=False, multiplicity="count"):
        if multiplicity not in ("count", "unit"):
            raise ValueError("multiplicity must be count or unit")
        regions = grid.regions()
        refined = tuple(refined_regions)
        if len(set(refined)) != len(refined) or any(type(x) is not int or not 0 <= x < len(regions) for x in refined):
            raise ValueError("invalid refined region identities")
        groups = []
        for i, region in enumerate(regions):
            if i in refined or (not pool_observed and region[0] < grid.frame_size):
                groups.extend((int(x),) for x in region)
            else:
                groups.append(tuple(map(int, region)))
        return cls(tuple(sorted(groups, key=lambda group: group[0])), grid, multiplicity)

    @property
    def original_length(self):
        return self.grid.length

    @property
    def packed_length(self):
        return len(self.groups)

    def validate(self, mask, ages=None):
        flat = [i for group in self.groups for i in group]
        if sorted(flat) != list(range(self.original_length)) or any(not group for group in self.groups):
            raise ValueError("pool groups must partition the original visual keys")
        if mask.dtype != torch.bool or mask.ndim != 2 or mask.shape[1] != self.original_length:
            raise ValueError("expected original visual-key visibility mask")
        if ages is not None and ages.shape != (self.original_length,):
            raise ValueError("cache ages must cover original visual positions")
        for group in self.groups:
            frames = {i // self.grid.frame_size for i in group}
            cameras = {(i % self.grid.width) // (self.grid.width // self.grid.cameras) for i in group}
            if len(frames) != 1 or len(cameras) != 1:
                raise ValueError("pool group crosses a frame or camera boundary")
            if not torch.equal(mask[:, group], mask[:, group[:1]].expand(-1, len(group))):
                raise ValueError("cannot pool keys with different native visibility")
            if ages is not None and not torch.equal(ages[list(group)], ages[group[0]].expand(len(group))):
                raise ValueError("mixed fresh/stale ages require explicit refresh or separate groups")

    def pack(self, key, value, mask, *, ages=None):
        """Mean native post-RoPE keys and raw values; no invented pooled RoPE.

        Returns packed K/V and additive mask. `count` adds log(number of source
        tokens), exact when keys within a group coincide; `unit` is unweighted
        mean pooling, a separate ablation. Neither is generally exact for unequal
        keys. Query positions never change.
        """
        self.validate(mask, ages)
        if key.shape != value.shape or key.ndim != 3 or key.shape[1] != self.original_length:
            raise ValueError("require [batch, original visual keys, width] K/V")
        if self.multiplicity not in ("count", "unit"):
            raise ValueError("unknown token multiplicity")
        packed_key = torch.stack([key[:, list(group)].mean(dim=1) for group in self.groups], dim=1)
        packed_value = torch.stack([value[:, list(group)].mean(dim=1) for group in self.groups], dim=1)
        bias = torch.stack([torch.full_like(mask[:, group[0]],
                     math.log(len(group)) if self.multiplicity == "count" else 0, dtype=torch.float32)
                     .masked_fill(~mask[:, group[0]], -torch.inf) for group in self.groups], dim=1)
        return packed_key, packed_value, bias
