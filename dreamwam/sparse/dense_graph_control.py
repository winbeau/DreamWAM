"""Native Dense transformer replay without an unused visual-cache export.

This is a benchmark control, not a policy approximation. It uses the same static
input/output buffering and graph scope as the visual-cache factors, but returns
only the original Joint outputs and current world-router diagnostics. Samplers,
RNG, preprocessors and schedulers are unchanged. Graph capture failure is fatal.
"""

from __future__ import annotations

from .visual_cache_graphs import GraphedVisualTokenCache, TensorTreeReplay


class NativeDenseGraph:
    refresh_every = 1
    keep_ratio = 1.0
    guidance_weight = 0.0
    graph_partial = False

    def __init__(self, model, *, graph_enabled=True, graph_warmup=3):
        if model.training or model.config.setting != "joint":
            raise ValueError("native Dense graph control requires Joint inference")
        self.model = model
        self.graph_enabled = graph_enabled
        self.graph_warmup = graph_warmup
        self.replay = None
        self._originals = []
        self._active = False
        self.last_stats = {}

    def __enter__(self):
        if self._originals or getattr(self.model, "_visual_ffn_context_cache", None):
            raise RuntimeError("another inference wrapper is already installed")
        mot = self.model.mot
        forward = mot.forward

        def compute(video_state, action_state):
            router = self.model.world_residual
            router.reset_runtime()
            result = forward(video_state=video_state, action_state=action_state,
                             residual_injection=router, sparse=None)
            return dict(result=result,
                        gates={name: list(values) for name, values in router._runtime_gates.items()},
                        previews={name: list(values) for name, values in router._runtime_local_preview_tokens.items()})

        def graph_forward(*, video_state, action_state, residual_injection,
                          sparse=None, step_index=0, num_steps=1):
            if not self._active:
                raise RuntimeError("native Dense graph requires a sample_action boundary")
            if residual_injection is not self.model.world_residual or (sparse is not None and sparse.enabled):
                raise ValueError("native Dense graph requires the unchanged Dense Joint path")
            request = dict(video_state=GraphedVisualTokenCache._state(video_state, video=True),
                           action_state=GraphedVisualTokenCache._state(action_state))
            if self.replay is None:
                self.replay = TensorTreeReplay(compute, request,
                    device=next(self.model.parameters()).device, enabled=self.graph_enabled,
                    warmup=self.graph_warmup)
            result = self.replay(request)
            router = self.model.world_residual
            router.reset_runtime()
            for name, values in result["gates"].items():
                router._runtime_gates[name].extend(values)
            for name, values in result["previews"].items():
                router._runtime_local_preview_tokens[name].extend(values)
            self._stats["dense_transformer_calls"] += 1
            self._stats["dense_graph_replays"] += int(self.replay.graph is not None)
            return result["result"]

        sample = self.model.sample_action
        def sample_action(*args, **kwargs):
            if self._active:
                raise RuntimeError("concurrent/reentrant model use is unsupported")
            self._active = True
            self._stats = dict(dense_transformer_calls=0, dense_graph_replays=0,
                               exported_visual_kv_layers=0)
            try:
                return sample(*args, **kwargs)
            finally:
                self.last_stats = dict(self._stats)
                self._active = False

        self._originals = [(mot, "forward", forward), (self.model, "sample_action", sample)]
        mot.forward = graph_forward
        self.model.sample_action = sample_action
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for owner, name, original in reversed(self._originals):
            setattr(owner, name, original)
        self._originals.clear()
        self._active = False

    def graph_stats(self):
        if self.replay is None:
            return {}
        return {"dense": dict(captured=self.replay.graph is not None,
                              capture_seconds=self.replay.capture_seconds,
                              replays=self.replay.replays)}

    def close_graphs(self):
        if self._active:
            raise RuntimeError("cannot release graphs during sampling")
        self.replay = None
