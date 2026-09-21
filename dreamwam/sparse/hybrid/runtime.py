"""One opt-in wrapper coordinating plans, selectors, math, and request state."""

from functools import wraps
from dataclasses import replace
import inspect
import time

import torch

from .config import HybridConfig
from .execution import HybridExecutor, selected_video_state, tensor_state
from .graphs import ExecutionDispatcher
from .schedule import StepContext, compile_plan, stable_hash
from .selection import Selection, select
from .state import VisualState
from .step_routing import observe_and_route


def scheduler_hash(model):
    return stable_hash({name: dict(type=type(scheduler).__qualname__,
                       num_train_timesteps=scheduler.num_train_timesteps,
                       shift=scheduler.shift, eps=scheduler.eps)
                       for name, scheduler in (("video", model.video_scheduler),
                                                ("action", model.action_scheduler))})


class HybridVisualRuntime:
    def __init__(self, model, config):
        self.config = config if isinstance(config, HybridConfig) else HybridConfig.from_mapping(config)
        self.base_config = self.config
        if model.training or model.config.setting != "joint":
            raise ValueError("hybrid visual execution requires Joint inference")
        self.model = model
        self.dispatch = ExecutionDispatcher(model, self.config)
        self.state = VisualState()
        self._originals = []
        self._active = False
        self._request_id = 0
        self.last_stats = {}

    def set_budget(self, query_ratio, read_ratio, *, chunk_extra_passes=None):
        """M1 may alter Q/KV and whole-chunk Q budgets between requests.

        Selection, schedule, reuse semantics and execution backend stay frozen.
        Graph dispatch remains bounded and keys include actual tensor shapes.
        """
        if self._active:
            raise RuntimeError("cannot change a chunk budget during sampling")
        if self.base_config.schedule.source == "profile":
            raise ValueError("cannot change a hash-frozen profile budget")
        self.config = replace(self.base_config, recompute_ratio=query_ratio, read_ratio=read_ratio,
            chunk_extra_passes=self.base_config.chunk_extra_passes if chunk_extra_passes is None else chunk_extra_passes)
        self.dispatch.config = self.config

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
        self.state.check_layout(current, frame_size, action_length)
        q_count, kv_count = self.config.budgets(length, frame_size)
        cap = self.config.query_row_cap(length, num_steps)
        if step_index == 0:
            compatibility = dict(video_length=length, tokens_per_frame=frame_size,
                                 scheduler_hash=scheduler_hash(self.model))
            self.plan = compile_plan(self.config, num_steps, compatibility=compatibility)
            self._stats["compatibility"] = compatibility
            self._stats["query_row_cap"] = cap
            if cap is not None and self.config.step_router is None:
                # A fixed M3 control must never silently change its schedule to
                # fit M1. Reject an infeasible allocation before any layer runs.
                demand = sum(length if op == "dense" else kv_count if self.config.reuse_mode == "structure"
                             else q_count if op == "sparse" else 0 for op in self.plan.schedule.operations)
                if demand > cap:
                    raise ValueError(f"fixed M3 schedule requires {demand} Q rows; chunk cap is {cap}")
            self.state.full_mask = self.model.mot.build_attention_mask(
                video_length=length, action_length=action_length,
                video_tokens_per_frame=frame_size, device=current.device)
            self.state.action_indices = torch.arange(action_length, device=current.device) + length
        decision = self.plan.decision(StepContext(step_index, num_steps))
        routing = None
        if self.config.step_router is not None:
            operation, routing = self._measure("step_routing", lambda: observe_and_route(
                self.config.step_router, current, self.state, step=step_index, num_steps=num_steps,
                query_rows=q_count, spent_rows=self._stats["query_rows_spent"], query_row_cap=cap,
                skip_exhausted_probe=self.config.chunk_extra_passes is not None))
            decision = replace(decision, operation=operation,
                read_mode="full" if operation == "dense" else self.config.read_mode,
                requested_q_ratio=1.0 if operation == "dense" else self.config.recompute_ratio if operation == "sparse" else 0.0,
                route_refresh=operation != "reuse", source="adaptive", reason=routing["reason"])
        effective = decision.operation
        if effective == "sparse" and q_count == length and kv_count == length:
            effective = "dense"
        self._effective_op = effective
        action_indices = self.state.action_indices
        selection = None
        if effective == "dense":
            result = self._call("dense", dict(video_state=tensor_state(video_state, video=True),
                                              action_state=tensor_state(action_state)))
            self._measure("cache_commit", lambda: self.state.commit_dense(result, current, step_index))
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
                                      chosen.video_key_probe_rows, chosen.video_query_probe_rows)
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
            q_rows, kv_rows = 0, self.state.route.numel()
        if effective != "reuse":
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
        self._stats["query_rows_spent"] += q_rows
        if routing is not None:
            self._stats["step_router_probe_rows"] += routing["probe_token_rows"]
            self._stats["step_router_seconds"] += routing["signal_seconds"]
        if cap is not None and self._stats["query_rows_spent"] > cap:
            raise AssertionError("execution exceeded its whole-chunk visual Q-row cap")
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
        if routing is not None:
            row.update(step_routing=routing, reference_schedule_op=self.plan.schedule.operations[step_index])
        if self.config.diagnostics == "trace":
            # Device values are serialized once at request end, not inside a timed step.
            row.update(route=self.state.route.clone(), updated_at=self.state.updated_at.clone(),
                       query=selection.query.clone() if selection else None,
                       video_t=self._times[0].detach(), action_t=self._times[1].detach())
        self._stats["steps"].append(row)
        return dict(video=self.state.output, action=result["action"])

    def __enter__(self):
        if self._originals or getattr(self.model, "_visual_ffn_context_cache", None):
            raise RuntimeError("another inference wrapper is already installed")
        self.model._visual_ffn_context_cache = self
        self.executor = HybridExecutor(self.model, self.model.mot.forward)
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
                               base_policy_hash=self.base_config.policy_hash,
                               query_ratio=self.config.recompute_ratio, read_ratio=self.config.read_ratio,
                               chunk_extra_passes=self.config.chunk_extra_passes, query_row_cap=None,
                               denoising_steps=0, dense_steps=0, sparse_steps=0, reuse_steps=0,
                               reuse_mode=self.config.reuse_mode,
                               action_layer_updates=0, computed_video_token_layers=0,
                               total_video_token_layers=0, read_video_token_layers=0,
                               action_probe_rows=0, video_key_probe_rows=0, video_query_probe_rows=0,
                               query_rows_spent=0, step_router_probe_rows=0, step_router_seconds=0.0,
                               graph_replays=0, steps=[], status="RUNNING")
            try:
                result = sample(*args, **kwargs)
                if self._stats["denoising_steps"] != self.plan.schedule.num_steps:
                    raise RuntimeError("sampler did not execute the complete plan")
                self._stats["status"] = "COMPLETE"
                cap = self._stats["query_row_cap"]
                self._stats["query_rows_unused"] = None if cap is None else cap - self._stats["query_rows_spent"]
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
        self.config = self.base_config
        self.dispatch.config = self.config

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
