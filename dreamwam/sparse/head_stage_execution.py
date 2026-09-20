"""Execute fixed M1 budgets with either a full mask or compact SDPA inputs.

This experimental controller does not change the policy adapter. Budgets are
immutable per layer/stage, conditioning and action rows retain their native
keys, and only future-video query rows use shortened per-head K/V tensors.
The compact full-budget control deliberately executes the same split path:
numerical drift and dispatch overhead must be measured even without pruning.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .head_stage_probe import stage_bounds
from .visual_cache_graphs import GraphedVisualTokenCache, TensorTreeReplay


def stage_index(step, num_steps):
    for index, (lo, hi) in enumerate(stage_bounds(num_steps)):
        if lo <= step < hi:
            return index
    raise ValueError("step outside the ten-step sampler")


def _heads(value, heads):
    batch, length, width = value.shape
    return value.reshape(batch, length, heads, width // heads).transpose(1, 2)


@dataclass
class BudgetPlan:
    mask: torch.Tensor
    fixed_mask: torch.Tensor
    groups: tuple
    budgets: tuple
    prefix: int
    video_length: int
    joint_length: int
    matrix_pairs: int


def make_plan(native, budgets, *, video_length, tokens_per_frame, block_size=14):
    """Build indices once, outside replay; reject non-native visibility."""
    if native.ndim == 4:
        if native.shape[:2] != (1, len(budgets)):
            raise ValueError("unexpected expanded mask layout")
        if not torch.equal(native, native[:, :1].expand_as(native)):
            raise ValueError("input mask already differs by head")
        native = native[0, 0]
    if native.ndim != 2 or native.dtype != torch.bool or native.shape[0] != native.shape[1]:
        raise ValueError("expected square native boolean mask")
    prefix, length = tokens_per_frame, native.shape[0]
    future = video_length - prefix
    if not 0 < prefix < video_length < length or block_size < 1 or future % block_size:
        raise ValueError("invalid complete-block video/action layout")
    blocks = future // block_size
    if not budgets or any(type(k) is not int or not 0 <= k <= blocks for k in budgets):
        raise ValueError("invalid integer future-block budgets")
    expected = torch.ones_like(native)
    expected[:prefix, prefix:] = False
    expected[prefix:video_length, video_length:] = False
    if not torch.equal(native, expected):
        raise ValueError("compact execution requires the native Joint visibility")
    mask = native[None, None].expand(1, len(budgets), -1, -1).clone()
    groups = []
    future_ids = torch.arange(future, device=native.device)
    for keep in sorted(set(budgets)):
        head_ids = [head for head, count in enumerate(budgets) if count == keep]
        chosen = (torch.linspace(0, blocks - 1, keep, device=native.device).round().long()
                  if keep else torch.empty(0, dtype=torch.long, device=native.device))
        membership = (future_ids[:, None] // block_size == chosen[None]).any(dim=1)
        keys = torch.cat((torch.arange(prefix, device=native.device),
                          prefix + future_ids[membership]))
        for head in head_ids:
            mask[0, head, prefix:video_length, prefix:video_length] &= membership[None]
        groups.append((torch.tensor(head_ids, device=native.device), keys,
                       len(head_ids) == len(budgets), keep == blocks))
    fixed = torch.cat((native[:prefix], native[video_length:]), dim=0)
    # Matrix extents submitted to SDPA, not FLOPs or hardware tile counters.
    pairs = (len(budgets) * (prefix + length - video_length) * length +
             future * sum(prefix + keep * block_size for keep in budgets))
    return BudgetPlan(mask, fixed, tuple(groups), tuple(budgets), prefix,
                      video_length, length, pairs)


def compact_attention(video_io, action_io, *, heads, plan):
    """Keep action softmax over all joint keys; shorten only future-video K/V."""
    qv, kv, vv = video_io[:3]
    qa, ka, va = action_io[:3]
    batch, length, width = qv.shape
    if length != plan.video_length or length + qa.shape[1] != plan.joint_length:
        raise ValueError("token layout changed")
    prefix = plan.prefix
    fixed = F.scaled_dot_product_attention(
        _heads(torch.cat((qv[:, :prefix], qa), dim=1), heads),
        _heads(torch.cat((kv, ka), dim=1), heads),
        _heads(torch.cat((vv, va), dim=1), heads), attn_mask=plan.fixed_mask)
    query, key, value = (_heads(t, heads) for t in (qv, kv, vv))
    future = None
    for head_ids, key_ids, all_heads, all_keys in plan.groups:
        q = query[:, :, prefix:]
        k, v = key, value
        if not all_heads:
            q, k, v = (t.index_select(1, head_ids) for t in (q, k, v))
        if not all_keys:
            k, v = (t.index_select(2, key_ids) for t in (k, v))
        current = F.scaled_dot_product_attention(q, k, v)
        if all_heads:
            future = current
        else:
            if future is None:
                future = torch.empty((batch, heads, length - prefix, width // heads),
                                     dtype=qv.dtype, device=qv.device)
            future.index_copy_(1, head_ids, current)
    return torch.cat((fixed[:, :, :prefix], future, fixed[:, :, prefix:]), dim=2).transpose(
        1, 2).reshape(batch, plan.joint_length, width)


class HeadStageExecution:
    def __init__(self, mot, profile, *, backend):
        if backend not in {"masked", "compact"}:
            raise ValueError("unknown HeadStage execution backend")
        if len(profile) != mot.num_layers or any(len(layer) != 3 for layer in profile):
            raise ValueError("profile must cover every layer and three stages")
        if any(len(row) != mot.num_heads or any(type(k) is not int or k < 0 for k in row)
               for layer in profile for row in layer):
            raise ValueError("invalid per-head profile")
        self.mot, self.backend = mot, backend
        self.profile = tuple(tuple(tuple(row) for row in layer) for layer in profile)
        self.plans = {}
        self.original = None

    def __enter__(self):
        if self.original is not None:
            raise RuntimeError("controller already installed")
        self.original = self.mot._joint_self_attention

        def attention(**kwargs):
            if kwargs["sparse"] is not None and kwargs["sparse"].enabled:
                raise ValueError("keep selection and execution experiments separate")
            stage = stage_index(kwargs["step_index"], kwargs["num_steps"])
            budgets = self.profile[kwargs["layer_index"]][stage]
            mask = kwargs["attention_mask"]
            key = (budgets, kwargs["video_length"], kwargs["tokens_per_frame"],
                   mask.shape[-1], mask.device)
            if key not in self.plans:
                self.plans[key] = make_plan(mask, budgets, video_length=kwargs["video_length"],
                                            tokens_per_frame=kwargs["tokens_per_frame"])
            plan = self.plans[key]
            if self.backend == "masked":
                return self.original(**{**kwargs, "attention_mask": plan.mask})
            return compact_attention(kwargs["video_io"], kwargs["action_io"],
                                     heads=self.mot.num_heads, plan=plan)

        self.mot._joint_self_attention = attention
        return self

    def __exit__(self, *args):
        self.mot._joint_self_attention = self.original
        self.original = None


class HeadStageGraph:
    """Three fixed transformer graphs; sampler, RNG and schedulers stay eager."""
    def __init__(self, model, *, enabled=True, warmup=3):
        if model.training or model.config.setting != "joint":
            raise ValueError("HeadStage graphs require Joint inference")
        self.model, self.enabled, self.warmup = model, enabled, warmup
        self.graphs, self.originals = {}, []
        self.active = False
        self.last_stats = {}

    def __enter__(self):
        if self.originals or getattr(self.model, "_visual_ffn_context_cache", None):
            raise RuntimeError("another inference wrapper is installed")
        model, mot = self.model, self.model.mot
        forward, sample = mot.forward, model.sample_action

        def graph_forward(*, video_state, action_state, residual_injection,
                          sparse=None, step_index=0, num_steps=1):
            if not self.active or residual_injection is not model.world_residual:
                raise RuntimeError("expected active native Joint sampler")
            if sparse is not None and sparse.enabled:
                raise ValueError("do not combine sparse routing factors")
            stage = stage_index(step_index, num_steps)
            request = dict(video_state=GraphedVisualTokenCache._state(video_state, video=True),
                           action_state=GraphedVisualTokenCache._state(action_state))
            if stage not in self.graphs:
                representative_step = stage_bounds(num_steps)[stage][0]

                def compute(video_state, action_state):
                    router = model.world_residual
                    router.reset_runtime()
                    result = forward(video_state=video_state, action_state=action_state,
                        residual_injection=router, sparse=None,
                        step_index=representative_step, num_steps=10)
                    return dict(result=result,
                        gates={k: list(v) for k, v in router._runtime_gates.items()},
                        previews={k: list(v) for k, v in router._runtime_local_preview_tokens.items()})

                self.graphs[stage] = TensorTreeReplay(compute, request,
                    device=next(model.parameters()).device, enabled=self.enabled, warmup=self.warmup)
            result = self.graphs[stage](request)
            router = model.world_residual
            router.reset_runtime()
            for k, values in result["gates"].items():
                router._runtime_gates[k].extend(values)
            for k, values in result["previews"].items():
                router._runtime_local_preview_tokens[k].extend(values)
            self.last_stats["stage_calls"][stage] += 1
            self.last_stats["graph_replays"] += int(self.graphs[stage].graph is not None)
            return result["result"]

        def sample_action(*args, **kwargs):
            if self.active:
                raise RuntimeError("concurrent inference unsupported")
            stage_bounds(kwargs.get("num_steps", 10))
            self.active = True
            self.last_stats = dict(stage_calls=[0, 0, 0], graph_replays=0)
            try:
                return sample(*args, **kwargs)
            finally:
                self.active = False

        self.originals = [(mot, "forward", forward), (model, "sample_action", sample)]
        mot.forward, model.sample_action = graph_forward, sample_action
        return self

    def __exit__(self, *args):
        for owner, name, original in reversed(self.originals):
            setattr(owner, name, original)
        self.originals.clear()
        self.active = False

    def graph_stats(self):
        return {stage: dict(captured=replay.graph is not None, replays=replay.replays,
                            capture_seconds=replay.capture_seconds)
                for stage, replay in self.graphs.items()}

    def close_graphs(self):
        if self.active:
            raise RuntimeError("cannot release active graphs")
        self.graphs.clear()
