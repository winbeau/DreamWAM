"""Offline instrumentation of real router probes and executed attention inputs.

The native attention output is unchanged. Probabilities are reconstructed from
the exact Q/K tensors supplied to that call; fused kernels do not expose their
internal probability tensor. Instrumented calls are never latency samples.
"""

import math

import torch

import dreamwam.mot as mot_module
from ..av import _to_heads
from . import execution, runtime as runtime_module, selection as selection_module
from .routing import project
from .execution import selected_video_state


def joint_probabilities(query, keys, mask, heads):
    q, k = _to_heads(query, heads), _to_heads(keys, heads)
    logits = (q * q.shape[-1] ** -0.5) @ k.transpose(-1, -2)
    if mask is not None:
        logits = logits.masked_fill(~mask.to(torch.bool), float("-inf"))
    return logits.float().softmax(-1)


class HybridTrace:
    """One eager request; include all heads/queries for the requested layers/steps."""

    def __init__(self, runtime, *, layers=(0,), steps=None):
        if runtime.config.backend != "eager" or runtime._active:
            raise ValueError("trace requires an idle eager hybrid runtime")
        if runtime.config.diagnostics != "trace":
            raise ValueError("trace requires diagnostics.level=trace for actual routes and ages")
        self.runtime = runtime
        self.layers = tuple(layers)
        self.steps = tuple(range(runtime.config.schedule.num_steps)) if steps is None else tuple(steps)
        for values, bound, label in ((self.layers, runtime.model.mot.num_layers, "layer"),
                                      (self.steps, runtime.config.schedule.num_steps, "step")):
            if not values or len(set(values)) != len(values) or any(type(i) is not int or i < 0 or i >= bound for i in values):
                raise ValueError("invalid or duplicate trace " + label)
        self.arrays, self.records, self.originals = {}, [], []
        self.grid = None
        self.current_selection = None
        self.operation = None
        self.layer_index = 0

    def _save(self, name, value):
        if name in self.arrays:
            raise RuntimeError("trace supports one request and one capture per key")
        self.arrays[name] = value.detach().cpu().numpy().copy()
        return name

    def _patch(self, owner, name, value):
        self.originals.append((owner, name, getattr(owner, name)))
        setattr(owner, name, value)

    def _probe(self, config, model, video, action, state, result, anchor):
        step = self.runtime._stats["denoising_steps"]
        if step not in self.steps or config.selection not in ("action", "action_context", "visual_context"):
            return
        length, actions = video["tokens"].shape[1], action["tokens"].shape[1]
        mot = model.mot
        key = project(model, video, "k")
        aio = mot._attention_input(mot.action_expert.blocks[0], action["tokens"],
                                  action["freqs"], action["time_modulation"])
        probabilities = joint_probabilities(aio[0], torch.cat((key, aio[1]), 1),
                                            state.full_mask[length:], mot.num_heads)
        direct = probabilities[..., :length].sum(-2).mean(1)
        prefix = f"s{step:02d}_probe"
        record = dict(kind="router_probe", layer=0, step=step, anchor=anchor,
            selection_method=config.selection, action_length=actions, video_length=length,
            joint_probabilities=self._save(prefix + "_joint", probabilities),
            direct=self._save(prefix + "_direct", direct),
            read_score=self._save(prefix + "_read", result.read),
            refresh_score=self._save(prefix + "_refresh", result.query),
            probe_qk="current full first-layer projection; not the packed keys used by execution",
            direct_used_by_selector=config.selection != "visual_context")
        if config.selection == "visual_context" or (config.selection == "action_context" and config.context_weight > 0):
            if config.selection == "visual_context":
                seeds = torch.arange(length, device=key.device)
                weights = torch.ones_like(direct) / length
            else:
                count = max(1, math.ceil(length * config.support_seed_ratio))
                seeds = direct.mean(0).argsort(descending=True, stable=True)[:count]
                weights = direct.index_select(1, seeds)
                weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
            query = project(model, selected_video_state(video, seeds), "q")
            vv = joint_probabilities(query, key, state.full_mask[:length, :length].index_select(0, seeds), mot.num_heads)
            support = (vv.mean(1) * weights[:, :, None]).sum(1)
            record.update(seeds=self._save(prefix + "_seeds", seeds),
                vv_probabilities=self._save(prefix + "_vv", vv),
                seed_weights=self._save(prefix + "_weights", weights),
                support=self._save(prefix + "_support", support))
        self.records.append(record)

    def __enter__(self):
        if self.originals or getattr(execution, "_trace_owner", None) is not None:
            raise RuntimeError("only one offline trace can be active per process")
        execution._trace_owner = self
        runtime = self.runtime
        original_pre = runtime.model.video_expert.pre_dit
        def pre(*args, **kwargs):
            result = original_pre(*args, **kwargs)
            grid = tuple(result["grid_size"])
            if self.grid is not None and grid != self.grid:
                raise ValueError("token grid changed during trace")
            self.grid = grid
            return result
        self._patch(runtime.model.video_expert, "pre_dit", pre)
        original_forward = runtime.model.mot.forward
        def forward(**kwargs):
            result = original_forward(**kwargs)
            step = kwargs["step_index"]
            row = runtime._stats["steps"][-1]
            query = row.get("query")
            if query is None:
                query = torch.empty(0, dtype=torch.long, device=runtime.state.route.device)
            self.records.append(dict(kind="executed_work", step=step,
                operation=row["effective_op"], requested_operation=row["requested_op"],
                reuse_mode=runtime.config.reuse_mode,
                query_ids=self._save(f"s{step:02d}_updated_queries", query),
                retained_key_ids=self._save(f"s{step:02d}_retained_keys", runtime.state.route),
                feature_ages=self._save(f"s{step:02d}_ages", step - runtime.state.updated_at)))
            return result
        self._patch(runtime.model.mot, "forward", forward)
        original_scores = selection_module.scores
        def scores(config, model, video, action, state, *, anchor=False):
            result = original_scores(config, model, video, action, state, anchor=anchor)
            self._probe(config, model, video, action, state, result, anchor)
            return result
        self._patch(selection_module, "scores", scores)
        original_select = runtime_module.select
        def select(*args, **kwargs):
            selected = original_select(*args, **kwargs)
            self.current_selection = selected
            return selected
        self._patch(runtime_module, "select", select)
        original_call = runtime._call
        def call(operation, request):
            self.operation, self.layer_index = operation, 0
            actions = request["action_state"]["tokens"].shape[1]
            length = math.prod(self.grid)
            if operation == "dense":
                self.executed_route = torch.arange(length, device=request["video_state"]["tokens"].device)
            elif operation in ("sparse", "fresh"):
                self.executed_route = self.current_selection.route
                if operation == "fresh" and runtime.plan.schedule.operations[runtime._stats["denoising_steps"]] == "reuse":
                    self.executed_route = runtime.state.route
            else:
                self.executed_route = runtime.state.route
            self.action_length = actions
            try:
                return original_call(operation, request)
            finally:
                self.operation = None
        self._patch(runtime, "_call", call)

        def wrap_attention(native):
            def attention(query, key, value, heads, mask=None):
                if self.operation is not None:
                    layer = self.layer_index
                    self.layer_index += 1
                    step = runtime._stats["denoising_steps"]
                    if layer in self.layers and step in self.steps:
                        actions = self.action_length
                        if key.shape[1] != self.executed_route.numel() + actions:
                            raise AssertionError("actual attention keys disagree with executed route")
                        probabilities = joint_probabilities(query[:, -actions:], key,
                            None if mask is None else mask[-actions:], heads)
                        prefix = f"s{step:02d}_l{layer:02d}_executed"
                        self.records.append(dict(kind="executed_attention", step=step, layer=layer,
                            operation=self.operation, action_length=actions,
                            video_length=math.prod(self.grid),
                            visual_key_ids=self._save(prefix + "_keys", self.executed_route),
                            joint_probabilities=self._save(prefix + "_joint", probabilities),
                            reconstruction="actual native-call Q/K, original mask, float32 softmax"))
                return native(query, key, value, heads, mask)
            return attention
        self._patch(execution, "scaled_dot_product_attention", wrap_attention(execution.scaled_dot_product_attention))
        self._patch(mot_module, "scaled_dot_product_attention", wrap_attention(mot_module.scaled_dot_product_attention))
        return self

    def __exit__(self, *args):
        for owner, name, original in reversed(self.originals):
            setattr(owner, name, original)
        self.originals.clear()
        if getattr(execution, "_trace_owner", None) is self:
            del execution._trace_owner
