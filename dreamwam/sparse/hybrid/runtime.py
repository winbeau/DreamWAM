"""One opt-in wrapper coordinating plans, selectors, math, and request state."""

from dataclasses import asdict
from functools import wraps
import inspect

import torch

from .config import HybridConfig
from .execution import HybridExecutor, selected_video_state, tensor_state
from .schedule import StepContext, compile_plan, stable_hash
from .selection import select
from .state import VisualState


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
        if self.config.read_mode != "full" or self.config.backend != "eager":
            raise ValueError("this implementation stage supports full-read eager execution only")
        self.model = model
        self.state = VisualState()
        self._originals = []
        self._active = False
        self._request_id = 0
        self.last_stats = {}

    def validate_num_steps(self, num_steps):
        return compile_plan(self.config, num_steps)

    def _call(self, operation, request):
        return getattr(self.executor, operation)(**request)

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
        if step_index == 0:
            compatibility = dict(video_length=length, tokens_per_frame=frame_size,
                                 scheduler_hash=scheduler_hash(self.model))
            self.plan = compile_plan(self.config, num_steps, compatibility=compatibility)
            self._stats["compatibility"] = compatibility
            self.state.full_mask = self.model.mot.build_attention_mask(
                video_length=length, action_length=action_length,
                video_tokens_per_frame=frame_size, device=current.device)
        decision = self.plan.decision(StepContext(step_index, num_steps))
        effective = decision.operation
        if effective == "sparse" and q_count == length and kv_count == length:
            effective = "dense"
        action_indices = torch.arange(action_length, device=current.device) + length
        selection = None
        if effective == "dense":
            result = self._call("dense", dict(video_state=tensor_state(video_state, video=True),
                                              action_state=tensor_state(action_state)))
            self.state.commit_dense(result, current, step_index)
            selection = select(self.config, self.model, video_state, action_state, self.state, anchor=True)
            self.state.pack(selection.route, step_index, full_read=self.config.read_mode == "full")
            q_rows, kv_rows = length, length
        elif effective == "sparse":
            selection = select(self.config, self.model, video_state, action_state, self.state)
            query, route = selection.query, selection.route
            rows = torch.cat((query, action_indices))
            columns = torch.cat((route, action_indices))
            mask = self.state.full_mask.index_select(0, rows).index_select(1, columns)
            kv = (self.state.kv if self.config.read_mode == "full" else
                  [{name: value.index_select(1, route) for name, value in layer.items()}
                   for layer in self.state.kv])
            result = self._call("sparse", dict(
                video_state=selected_video_state(video_state, query),
                action_state=tensor_state(action_state), kv=kv,
                query_slots=torch.searchsorted(route, query), mask=mask))
            self.state.commit_sparse(result, current, query, route, step_index,
                                     full_read=self.config.read_mode == "full")
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
        self._stats["action_probe_rows"] += selection.action_probe_rows if selection else 0
        row = dict(step_index=step_index, requested_op=decision.operation, effective_op=effective,
                   q_rows=q_rows, kv_rows=kv_rows, read_mode=decision.read_mode,
                   route_refresh=effective != "reuse", route_age=step_index - self.state.route_step,
                   action_layer_updates=layers, fallback_reason=None,
                   action_probe_rows=selection.action_probe_rows if selection else 0)
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
            self.state.clear()
            self._active = True
            self._request_id += 1
            self._stats = dict(request_id=self._request_id, plan_hash=self.plan.plan_hash,
                               denoising_steps=0, dense_steps=0, sparse_steps=0, reuse_steps=0,
                               action_layer_updates=0, computed_video_token_layers=0,
                               total_video_token_layers=0, read_video_token_layers=0,
                               action_probe_rows=0, steps=[], status="RUNNING")
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

    def __exit__(self, exc_type, exc_value, traceback):
        for owner, name, original in reversed(self._originals):
            setattr(owner, name, original)
        self._originals.clear()
        self.state.clear()
        self._active = False
        if getattr(self.model, "_visual_ffn_context_cache", None) is self:
            del self.model._visual_ffn_context_cache
