"""Offline causal VV interventions for M1, with all action rows unchanged.

This is a diagnostic, not a fast inference backend. It keeps the full fused
joint attention call and changes only one head's future-query VV mask in one
layer and denoising stage. The matched full-budget control uses exactly the
same expanded mask layout. It never prunes action rows or the conditioning
frame, and never enables temporal caching or sparse routing heuristics.
"""

from __future__ import annotations

from functools import wraps
import math

import torch


def stage_bounds(num_steps):
    if num_steps != 10:
        raise ValueError("this offline protocol fixes ten steps and stages [0,3), [3,7), [7,10)")
    return ((0, 3), (3, 7), (7, 10))


def intervention_mask(native, *, num_heads, video_length, tokens_per_frame,
                      head=None, future_keep_ratio=1.0, block_size=14):
    """Full joint masks; evenly spaced future blocks at the requested budget."""
    if native.ndim != 2 or native.dtype != torch.bool or native.shape[0] != native.shape[1]:
        raise ValueError("expected the native square boolean joint mask")
    if not 0 < tokens_per_frame < video_length < native.shape[0] or num_heads < 1:
        raise ValueError("invalid action/video layout")
    if not 0 <= future_keep_ratio <= 1 or block_size < 1:
        raise ValueError("invalid future budget or block size")
    if head is not None and not 0 <= head < num_heads:
        raise ValueError("head outside model")
    mask = native[None, None].expand(1, num_heads, -1, -1).clone()
    future_length = video_length - tokens_per_frame
    blocks = math.ceil(future_length / block_size)
    keep = math.ceil(blocks * future_keep_ratio)
    if head is None or keep == blocks:
        return mask, dict(future_blocks=blocks, kept_future_blocks=blocks,
                          target_vv_density=1.0, removed_pairs=0)
    selected = (torch.linspace(0, blocks - 1, keep, device=native.device).round().long()
                if keep else torch.empty(0, dtype=torch.long, device=native.device))
    indices = torch.arange(future_length, device=native.device)
    membership = (indices[:, None] // block_size == selected[None]).any(dim=1)
    mask[0, head, tokens_per_frame:video_length, tokens_per_frame:video_length] &= membership[None]
    # This offline metadata may synchronize; no speedup is attributed to it.
    original_pairs = int(native[:video_length, :video_length].sum())
    kept_pairs = int(mask[0, head, :video_length, :video_length].sum())
    return mask, dict(future_blocks=blocks, kept_future_blocks=keep,
                      target_vv_density=kept_pairs / original_pairs,
                      removed_pairs=original_pairs - kept_pairs)


class HeadStageProbe:
    def __init__(self, model):
        self.model = model
        self.originals = []
        self.target = None
        self.masks = {}
        self.last_video = None
        self.last_raw_action = None
        self.stats = {}

    def configure(self, *, layer=None, head=None, stage=None, future_keep_ratio=1.0):
        if any(value is not None for value in (layer, head, stage)):
            if not (isinstance(layer, int) and 0 <= layer < self.model.mot.num_layers and
                    isinstance(head, int) and 0 <= head < self.model.mot.num_heads and
                    isinstance(stage, int) and 0 <= stage < 3):
                raise ValueError("supply a valid layer, head and stage together")
            self.target = (layer, head, stage, float(future_keep_ratio))
        else:
            self.target = None
        if not 0 <= future_keep_ratio <= 1:
            raise ValueError("invalid future budget")
        self.masks.clear()

    def __enter__(self):
        if self.originals or getattr(self.model, "_visual_ffn_context_cache", None):
            raise RuntimeError("M1 probing requires an unwrapped native model")
        mot = self.model.mot
        attention, sample = mot._joint_self_attention, self.model.sample_action
        video_step = self.model.video_scheduler.step

        @wraps(attention)
        def probe_attention(**kwargs):
            if kwargs["sparse"] is not None and kwargs["sparse"].enabled:
                raise ValueError("M1 diagnostic cannot combine sparse selection factors")
            bounds = stage_bounds(kwargs["num_steps"])
            active = False
            if self.target is not None:
                layer, head, stage, ratio = self.target
                lo, hi = bounds[stage]
                active = kwargs["layer_index"] == layer and lo <= kwargs["step_index"] < hi
            key = bool(active)
            if key not in self.masks:
                mask, metadata = intervention_mask(kwargs["attention_mask"],
                    num_heads=mot.num_heads, video_length=kwargs["video_length"],
                    tokens_per_frame=kwargs["tokens_per_frame"],
                    head=self.target[1] if active else None,
                    future_keep_ratio=self.target[3] if active else 1.0)
                self.masks[key] = (mask, metadata)
            mask, metadata = self.masks[key]
            self.stats["attention_calls"] += 1
            if active:
                self.stats["target_calls"] += 1
                self.stats["target_steps"].append(kwargs["step_index"])
                self.stats["mask"] = metadata
            return attention(**{**kwargs, "attention_mask": mask})

        @wraps(video_step)
        def observe_video_step(*args, **kwargs):
            result = video_step(*args, **kwargs)
            self.last_video = result
            self.stats["video_scheduler_steps"] += 1
            return result

        @wraps(sample)
        def probe_sample(*args, **kwargs):
            stage_bounds(kwargs.get("num_steps", 10))
            self.masks.clear()
            self.last_video = self.last_raw_action = None
            self.stats = dict(attention_calls=0, target_calls=0, target_steps=[], video_scheduler_steps=0)
            try:
                result = sample(*args, **kwargs)
                self.last_raw_action = result.detach().clone()
                # The sampler restores the conditioned frame in this same tensor.
                self.last_video = self.last_video.detach().clone()
                return result
            except BaseException:
                self.last_video = self.last_raw_action = None
                raise
            finally:
                self.masks.clear()

        self.originals = [(mot, "_joint_self_attention", attention),
                          (self.model, "sample_action", sample),
                          (self.model.video_scheduler, "step", video_step)]
        mot._joint_self_attention = probe_attention
        self.model.sample_action = probe_sample
        self.model.video_scheduler.step = observe_video_step
        return self

    def __exit__(self, *args):
        for owner, name, original in reversed(self.originals):
            setattr(owner, name, original)
        self.originals.clear()
        self.masks.clear()
        self.last_video = self.last_raw_action = None
