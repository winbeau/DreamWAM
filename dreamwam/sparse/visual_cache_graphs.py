"""An isolated dispatch factor on the existing action-guided visual cache.

By default, only dense Joint and cached-video action transformer calls are
captured. ``graph_partial=True`` is a separate factor that also captures partial
refresh math. Token selection, pre/post-DiT, both schedulers and the original CPU
RNG stay eager. The full-budget control captures every dense call.
All inputs are staged for every invocation, and every output (including visual
K/V and world-router diagnostics) is copied out before reuse. Capture failures
are fatal: an eager fallback must never be timed as graph replay.

``graph_enabled=False`` is an explicit buffering control, useful on CPU as well.
Graphs persist across context entries; ``close_graphs`` releases their buffers.
The policy enables this class only through an explicit visual_cache graph option.
"""

from __future__ import annotations

import time

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten

from .action_guided_visual_token_cache import ActionGuidedVisualTokenCache


class TensorTreeReplay:
    """Fixed tensor inputs, immutable metadata, and non-aliasing tree outputs."""

    def __init__(self, function, request, *, device, enabled=True, warmup=3):
        self.function = function
        self.device = torch.device(device)
        self.enabled = enabled
        if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 1:
            raise ValueError("graph warmup must be a positive integer")
        if enabled and self.device.type != "cuda":
            raise ValueError("CUDA graph replay requires a CUDA device")
        self.warmup = warmup
        leaves, self.spec = tree_flatten(request)
        self.signature = [self._signature(value) for value in leaves]
        self.buffers = [value.detach().to(self.device).clone() if isinstance(value, torch.Tensor)
                        else value for value in leaves]
        self.graph = None
        self.output = None
        self.capture_seconds = 0.0
        self.replays = 0

    @staticmethod
    def _signature(value):
        if isinstance(value, torch.Tensor):
            return ("tensor", tuple(value.shape), value.dtype, value.device, tuple(value.stride()))
        if value is not None and type(value) not in (str, int, float, bool):
            raise TypeError(f"unsupported static metadata: {type(value).__name__}")
        return (type(value), value)

    def _fill(self, request):
        leaves, spec = tree_flatten(request)
        if spec != self.spec or [self._signature(value) for value in leaves] != self.signature:
            raise ValueError("graph input structure, shape, dtype, device, stride or metadata changed")
        for buffer, value in zip(self.buffers, leaves):
            if isinstance(buffer, torch.Tensor):
                buffer.copy_(value)

    def _compute(self):
        return self.function(**tree_unflatten(self.buffers, self.spec))

    def __call__(self, request):
        if torch.is_grad_enabled():
            raise RuntimeError("tensor replay is inference-only; use no_grad")
        self._fill(request)
        if self.enabled and self.graph is None:
            started = time.perf_counter()
            stream = torch.cuda.Stream(device=self.device)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(stream):
                for _ in range(self.warmup):
                    # Partial refresh mutates static K/V inputs. Every warmup
                    # must start from the supplied anchor, just like replay.
                    self._fill(request)
                    self._compute()
            torch.cuda.current_stream(self.device).wait_stream(stream)
            torch.cuda.synchronize(self.device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                self.output = self._compute()
            self.graph = graph
            self.capture_seconds = time.perf_counter() - started
            self._fill(request)
        if self.graph is not None:
            self.graph.replay()
            self.replays += 1
            output = self.output
        else:
            output = self._compute()
        leaves, spec = tree_flatten(output)
        # Callers may scatter into K/V, so graph-owned outputs cannot escape.
        return tree_unflatten([value.clone() if isinstance(value, torch.Tensor) else value
                               for value in leaves], spec)


class GraphedVisualTokenCache(ActionGuidedVisualTokenCache):
    def __init__(self, model, *, keep_ratio, refresh_every=5, guidance_weight=1.0,
                 graph_enabled=True, graph_warmup=3, graph_partial=False):
        super().__init__(model, keep_ratio=keep_ratio, refresh_every=refresh_every,
                         guidance_weight=guidance_weight)
        self.graph_enabled = graph_enabled
        self.graph_warmup = graph_warmup
        self.graph_partial = graph_partial
        self.graphs = {}

    def _begin(self):
        super()._begin()
        self._stats.update(dense_buffered_calls=0, action_buffered_calls=0,
                           partial_buffered_calls=0, dense_graph_replays=0,
                           action_graph_replays=0, partial_graph_replays=0)

    def _call(self, name, function, request):
        if name not in self.graphs:
            self.graphs[name] = TensorTreeReplay(
                function, request, device=next(self.model.parameters()).device,
                enabled=self.graph_enabled, warmup=self.graph_warmup,
            )
        replay = self.graphs[name]
        result = replay(request)
        self._stats[f"{name}_buffered_calls"] += 1
        self._stats[f"{name}_graph_replays"] += int(replay.graph is not None)
        return result

    @staticmethod
    def _state(state, *, video=False):
        keys = ("tokens", "freqs", "time_modulation", "context", "context_mask")
        if video:
            keys += ("tokens_per_frame",)
        return {key: state[key] for key in keys}

    def _refresh(self, forward, **kwargs):
        def graph_forward(*, video_state, action_state, residual_injection,
                          sparse=None, step_index=0, num_steps=1):
            if residual_injection is not self.model.world_residual:
                raise ValueError("graph factor requires the unchanged Joint world router")
            if sparse is not None and sparse.enabled:
                raise ValueError("graph factor cannot be combined with sparse attention")

            def compute(video_state, action_state):
                router = self.model.world_residual
                router.reset_runtime()
                result = forward(video_state=video_state, action_state=action_state,
                                 residual_injection=router, sparse=None)
                # _capture is true here: the existing attention hook owns K/V.
                return dict(result=result,
                            kv=[dict(self.video_kv[layer]) for layer in range(self.model.mot.num_layers)],
                            gates={key: list(values) for key, values in router._runtime_gates.items()},
                            previews={key: list(values) for key, values in router._runtime_local_preview_tokens.items()})

            result = self._call("dense", compute, dict(
                video_state=self._state(video_state, video=True),
                action_state=self._state(action_state),
            ))
            # Python hooks do not execute on replay. Restore their current-call
            # products explicitly, never the preceding request's cached tensors.
            self.video_kv = dict(enumerate(result["kv"]))
            router = self.model.world_residual
            router.reset_runtime()
            for key, values in result["gates"].items():
                router._runtime_gates[key].extend(values)
            for key, values in result["previews"].items():
                router._runtime_local_preview_tokens[key].extend(values)
            return result["result"]

        return super()._refresh(graph_forward, **kwargs)

    def _reuse(self, *, video_state, action_state):
        mot = self.model.mot
        original = mot.forward_action_with_video_cache

        def graph_action(**kwargs):
            kwargs["action_state"] = self._state(kwargs["action_state"])
            return self._call("action", original, kwargs)

        mot.forward_action_with_video_cache = graph_action
        try:
            return super()._reuse(video_state=video_state, action_state=action_state)
        finally:
            mot.forward_action_with_video_cache = original

    def _partial_refresh(self, video_state, action_state, residual_injection, index, video):
        if not self.graph_partial:
            return super()._partial_refresh(video_state, action_state, residual_injection, index, video)
        if residual_injection is not self.model.world_residual:
            raise ValueError("graph factor requires the unchanged Joint world router")
        original = super()._partial_refresh

        def compute(video_state, action_state, index, video, kv):
            router = self.model.world_residual
            router.reset_runtime()
            current_cache = self.video_kv
            self.video_kv = dict(enumerate(kv))
            try:
                action, refreshed = original(video_state, action_state, router, index, video)
                return dict(action=action, video=refreshed, kv=kv,
                            gates={key: list(values) for key, values in router._runtime_gates.items()},
                            previews={key: list(values) for key, values in router._runtime_local_preview_tokens.items()})
            finally:
                self.video_kv = current_cache

        result = self._call("partial", compute, dict(
            video_state=self._state(video_state, video=True),
            action_state=self._state(action_state), index=index, video=video,
            kv=[self.video_kv[layer] for layer in range(self.model.mot.num_layers)],
        ))
        self.video_kv = dict(enumerate(result["kv"]))
        router = self.model.world_residual
        router.reset_runtime()
        for key, values in result["gates"].items():
            router._runtime_gates[key].extend(values)
        for key, values in result["previews"].items():
            router._runtime_local_preview_tokens[key].extend(values)
        return result["action"], result["video"]

    def graph_stats(self):
        return {name: dict(captured=replay.graph is not None,
                           capture_seconds=replay.capture_seconds, replays=replay.replays)
                for name, replay in self.graphs.items()}

    def close_graphs(self):
        if self._active:
            raise RuntimeError("cannot release graphs during a sampling request")
        self.graphs.clear()
