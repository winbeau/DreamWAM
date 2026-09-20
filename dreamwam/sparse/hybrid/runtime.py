"""One opt-in wrapper coordinating plans, selectors, math, and request state."""

from functools import wraps
import inspect
import time

import torch

from .config import HybridConfig
from .execution import HybridExecutor, selected_video_state, tensor_state
from .graphs import ExecutionDispatcher
from .schedule import StepContext, compile_plan, stable_hash
from .selection import Selection, select
from .state import VisualState
from ..profile.geometry import TokenGrid
from .native_packing import compile_pool_geometry
from .native_scores import score_work, scoring_heads


def scheduler_hash(model):
    return stable_hash({name: dict(type=type(scheduler).__qualname__,
                       num_train_timesteps=scheduler.num_train_timesteps,
                       shift=scheduler.shift, eps=scheduler.eps)
                       for name, scheduler in (("video", model.video_scheduler),
                                                ("action", model.action_scheduler))})


class HybridVisualRuntime:
    def __init__(self, model, config):
        self.config = config if isinstance(config, HybridConfig) else HybridConfig.from_mapping(config)
        if model.training or model.config.setting != "joint":
            raise ValueError("hybrid visual execution requires Joint inference")
        if self.config.native_routing:
            self.config.native_routing.sampled_layers(model.mot.num_layers)
        self.model = model
        self.dispatch = ExecutionDispatcher(model, self.config)
        self.state = VisualState()
        self._originals = []
        self._active = False
        self._request_id = 0
        self.last_stats = {}

    def validate_num_steps(self, num_steps):
        return compile_plan(self.config, num_steps)

    def _call(self, operation, request):
        return self._measure("execute_" + operation,
                             lambda: self.dispatch(operation, getattr(self.executor, operation), request))

    def _measure(self, phase, function):
        if self.config.diagnostics != "trace":
            return function()
        device = next(self.model.parameters()).device
        start_event = end_event = None
        if device.type == "cuda":
            start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start_event.record()
        started = time.perf_counter()
        try:
            return function()
        finally:
            elapsed = time.perf_counter() - started
            if end_event is not None:
                end_event.record()
            self._phases.append((phase, self._stats["denoising_steps"], elapsed, start_event, end_event))

    def _sparse_request(self, video_state, action_state, selection):
        query, route = selection.query, selection.route
        rows = torch.cat((query, self.state.action_indices))
        columns = torch.cat((route, self.state.action_indices))
        mask = self.state.full_mask.index_select(0, rows).index_select(1, columns)
        kv = (self.state.kv if self.config.read_mode == "full" else
              [{name: value.index_select(1, route) for name, value in layer.items()} for layer in self.state.kv])
        return dict(video_state=selected_video_state(video_state, query),
                    action_state=tensor_state(action_state), kv=kv,
                    query_slots=torch.searchsorted(route, query), mask=mask)

    def _restore_router(self, result):
        if "gates" not in result:
            return
        router = self.model.world_residual
        router.reset_runtime()
        for name, values in result["gates"].items():
            router._runtime_gates[name].extend(values)
        for name, values in result["previews"].items():
            router._runtime_local_preview_tokens[name].extend(values)

    def _forward(self, *, video_state, action_state, residual_injection,
                 sparse=None, step_index=0, num_steps=1):
        if not self._active or step_index != self._stats["denoising_steps"]:
            raise RuntimeError("hybrid requires sequential sample_action steps")
        if residual_injection is not self.model.world_residual or (sparse is not None and sparse.enabled):
            raise ValueError("hybrid cannot be combined with another router or sparse wrapper")
        current = video_state["tokens"]
        batch, length = current.shape[:2]
        frame_size = video_state["tokens_per_frame"]
        action_length = action_state["tokens"].shape[1]
        native = self.config.native_routing
        self.state.check_layout(current, frame_size, action_length)
        q_count, kv_count = self.config.budgets(length, frame_size)
        if step_index == 0:
            compatibility = dict(video_length=length, tokens_per_frame=frame_size,
                                 scheduler_hash=scheduler_hash(self.model))
            self.plan = compile_plan(self.config, num_steps, compatibility=compatibility)
            self._stats["compatibility"] = compatibility
            mask_device = "cpu" if native and native.packing == "pool" else current.device
            full_mask = self.model.mot.build_attention_mask(
                video_length=length, action_length=action_length,
                video_tokens_per_frame=frame_size, device=mask_device)
            if native and native.packing == "pool":
                self.state.native_geometry = compile_pool_geometry(TokenGrid(*video_state["grid_size"]),
                    full_mask, native.refine_fraction, current.device)
                if self.state.native_geometry["packed_length"] != kv_count:
                    raise ValueError("explicit read count does not match the pooled geometry budget")
            self.state.full_mask = full_mask.to(current.device)
            self.state.action_indices = torch.arange(action_length, device=current.device) + length
        decision = self.plan.decision(StepContext(step_index, num_steps))
        effective = decision.operation
        if effective == "sparse" and q_count == length and kv_count == length:
            effective = "dense"
        action_indices = self.state.action_indices
        selection = None
        if effective == "dense":
            result = self._call("dense", dict(video_state=tensor_state(video_state, video=True),
                                              action_state=tensor_state(action_state)))
            self._measure("cache_commit", lambda: self.state.commit_dense(result, current, step_index))
            if native and native.separate_packing:
                packed = self._call("pack_native", dict(kv=result["kv"], scores=result["native_scores"],
                    valid=result["native_valid"], full_mask=self.state.full_mask, grid_size=video_state["grid_size"],
                    read_count=kv_count, geometry=self.state.native_geometry))
                self._measure("cache_commit", lambda: self.state.pack_native(packed, step_index))
                selection = Selection(torch.arange(length, device=current.device), packed["route"], fallback=packed["fallback"])
            else:
                selection = self._measure("selection", lambda: select(
                    self.config, self.model, video_state, action_state, self.state, anchor=True))
                self._measure("pack", lambda: self.state.pack(
                    selection.route, step_index, full_read=self.config.read_mode == "full"))
            q_rows, kv_rows = length, length
        elif self.config.reuse_mode == "structure":
            if effective == "sparse":
                chosen = self._measure("selection", lambda: select(
                    self.config, self.model, video_state, action_state, self.state, anchor=True))
                selection = Selection(chosen.route, chosen.route, chosen.action_probe_rows,
                                      chosen.video_key_probe_rows, chosen.video_query_probe_rows, chosen.fallback)
            else:
                selection = Selection(self.state.route, self.state.route)
            route = selection.route
            def prepare_fresh():
                positions = torch.cat((route, action_indices))
                return dict(video_state=selected_video_state(video_state, route),
                            action_state=tensor_state(action_state),
                            mask=self.state.full_mask.index_select(0, positions).index_select(1, positions))
            result = self._call("fresh", self._measure("prepare", prepare_fresh))
            self._measure("cache_commit", lambda: self.state.commit_fresh(
                result, current, route, step_index, refresh=effective == "sparse"))
            q_rows = kv_rows = route.numel()
        elif effective == "sparse":
            selection = self._measure("selection", lambda: select(
                self.config, self.model, video_state, action_state, self.state))
            query, route = selection.query, selection.route
            request = self._measure("prepare", lambda: self._sparse_request(video_state, action_state, selection))
            result = self._call("sparse", request)
            self._measure("cache_commit", lambda: self.state.commit_sparse(
                result, current, query, route, step_index, full_read=self.config.read_mode == "full"))
            q_rows, kv_rows = query.numel(), route.numel()
        else:
            if self.state.output is None:
                raise RuntimeError("reuse requires a current-request dense anchor")
            result = self._call("reuse", dict(action_state=tensor_state(action_state),
                                              kv=self.state.packed, mask=self.state.action_mask))
            q_rows, kv_rows = 0, self.state.route.shape[-1]
        if effective != "reuse" and not (native and native.separate_packing):
            columns = torch.cat((self.state.route, action_indices))
            self.state.action_mask = self.state.full_mask[length:].index_select(1, columns)
        self._restore_router(result)
        layers = self.model.mot.num_layers
        self._stats["denoising_steps"] += 1
        self._stats[f"{effective}_steps"] += 1
        self._stats["action_layer_updates"] += layers
        self._stats["computed_video_token_layers"] += batch * q_rows * layers
        self._stats["total_video_token_layers"] += batch * length * layers
        self._stats["read_video_token_layers"] += batch * kv_rows * layers
        self._stats["action_probe_rows"] += selection.action_probe_rows if selection else 0
        self._stats["video_key_probe_rows"] += selection.video_key_probe_rows if selection else 0
        self._stats["video_query_probe_rows"] += selection.video_query_probe_rows if selection else 0
        self._stats["graph_replays"] += int(self.dispatch.last_replayed)
        row = dict(step_index=step_index, requested_op=decision.operation, effective_op=effective,
                   q_rows=q_rows, kv_rows=kv_rows, read_mode=decision.read_mode,
                   reuse_mode=self.config.reuse_mode,
                   reused_visual_features=effective == "reuse" and self.config.reuse_mode == "features",
                   route_refresh=effective != "reuse", route_age=step_index - self.state.route_step,
                   action_layer_updates=layers, fallback_reason=None,
                   action_probe_rows=selection.action_probe_rows if selection else 0,
                   video_key_probe_rows=selection.video_key_probe_rows if selection else 0,
                   video_query_probe_rows=selection.video_query_probe_rows if selection else 0,
                   graph_key=self.dispatch.last_key, graph_replay=self.dispatch.last_replayed)
        if native:
            work = score_work(native, layers=len(self.executor.native_layers) if effective == "dense" else 0,
                              num_heads=self.model.mot.num_heads, video_length=length, action_length=action_length)
            for key, count in work.items():
                self._stats[key] += count
            row.update(**work, native_score_step=self.state.native_score_step,
                native_score_age=step_index - self.state.native_score_step,
                native_score_layers=list(self.executor.native_layers) if effective == "dense" else [],
                native_score_heads=list(scoring_heads(self.model.mot.num_heads, native.heads)) if effective == "dense" else [],
                read_layout="layerwise" if native.layerwise else "shared", packing=native.packing,
                represented_visual_rows=length if native.packing == "pool" else kv_count,
                selector_fallback=None)
            if selection is not None and selection.fallback is not None:
                row["_native_fallback"] = selection.fallback.clone()
        if self.config.diagnostics == "trace":
            # Device values are serialized once at request end, not inside a timed step.
            row.update(route=self.state.route.clone(), updated_at=self.state.updated_at.clone(),
                       query=selection.query.clone() if selection else None,
                       video_t=self._times[0].detach(), action_t=self._times[1].detach())
            if native and native.separate_packing:
                row.update(read_group_members=self.state.read_members.clone(), read_group_sizes=self.state.read_sizes.clone())
        self._stats["steps"].append(row)
        return dict(video=self.state.output, action=result["action"])

    def __enter__(self):
        if self._originals or getattr(self.model, "_visual_ffn_context_cache", None):
            raise RuntimeError("another inference wrapper is already installed")
        self.model._visual_ffn_context_cache = self
        self.executor = HybridExecutor(self.model, self.model.mot.forward, self.config.native_routing)
        sample = self.model.sample_action
        signature = inspect.signature(sample)

        @wraps(sample)
        def sample_action(*args, **kwargs):
            if self._active:
                raise RuntimeError("concurrent/reentrant sampling is unsupported")
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            self.plan = self.validate_num_steps(bound.arguments["num_steps"])
            self.dispatch.begin_request()
            self.state.clear()
            self._active = True
            self._phases = []
            self._request_id += 1
            self._stats = dict(request_id=self._request_id, plan_hash=self.plan.plan_hash,
                               denoising_steps=0, dense_steps=0, sparse_steps=0, reuse_steps=0,
                               reuse_mode=self.config.reuse_mode,
                               action_layer_updates=0, computed_video_token_layers=0,
                               total_video_token_layers=0, read_video_token_layers=0,
                               action_probe_rows=0, video_key_probe_rows=0, video_query_probe_rows=0,
                               graph_replays=0, steps=[], status="RUNNING")
            if self.config.native_routing:
                self._stats.update(native_score_query_head_rows=0, native_score_value_head_rows=0,
                                   selector_fallback_events=0)
            try:
                result = sample(*args, **kwargs)
                if self._stats["denoising_steps"] != self.plan.schedule.num_steps:
                    raise RuntimeError("sampler did not execute the complete plan")
                self._stats["status"] = "COMPLETE"
                return result
            except BaseException:
                self._stats["status"] = "ERROR"
                raise
            finally:
                try:
                    if self.config.diagnostics == "trace":
                        phases = []
                        for phase, index, seconds, start, end in self._phases:
                            if end is not None:
                                end.synchronize()
                            phases.append(dict(phase=phase, step_index=index, cpu_seconds=seconds,
                                cuda_stream_span_seconds=start.elapsed_time(end) / 1000 if end else None))
                        self._stats["phase_timings"] = phases
                    native_rows = [row for row in self._stats["steps"] if "_native_fallback" in row]
                    if native_rows:
                        events = torch.stack([row.pop("_native_fallback") for row in native_rows]).cpu().tolist()
                        self._stats["selector_fallback_events"] = sum(events)
                        for row, event in zip(native_rows, events):
                            row["selector_fallback"] = event
                    for row in self._stats["steps"]:
                        if "route" in row:
                            route = row["route"].cpu().tolist()
                            age = row["step_index"] - row.pop("updated_at").cpu()
                            row["route"], row["route_hash"] = route, stable_hash(route)
                            row["feature_age_max"] = int(age.max())
                            row["feature_age_mean"] = float(age.float().mean())
                            for name in ("query", "video_t", "action_t"):
                                value = row[name]
                                row[name] = value.cpu().tolist() if value is not None else None
                            for name in ("read_group_members", "read_group_sizes"):
                                if name in row:
                                    row[name] = row[name].cpu().tolist()
                    self.last_stats = dict(self._stats)
                finally:
                    self.state.clear()
                    self._times = None
                    self._phases = []
                    self._active = False

        conditioned = self.model._forward_conditioned

        @wraps(conditioned)
        def forward_conditioned(*args, **kwargs):
            self._times = (kwargs["video_timestep"], kwargs["action_timestep"])
            return conditioned(*args, **kwargs)

        self._originals = [(self.model.mot, "forward", self.model.mot.forward),
                           (self.model, "sample_action", sample),
                           (self.model, "_forward_conditioned", conditioned)]
        self.model.mot.forward = self._forward
        self.model.sample_action = sample_action
        self.model._forward_conditioned = forward_conditioned
        return self

    def reset(self):
        if self._active:
            raise RuntimeError("cannot reset during sampling")
        self.state.clear()

    def close_graphs(self):
        self.reset()
        self.dispatch.clear()

    def graph_stats(self):
        return self.dispatch.stats()

    def __exit__(self, exc_type, exc_value, traceback):
        for owner, name, original in reversed(self._originals):
            setattr(owner, name, original)
        self._originals.clear()
        self.state.clear()
        self._active = False
        if getattr(self.model, "_visual_ffn_context_cache", None) is self:
            del self.model._visual_ffn_context_cache
