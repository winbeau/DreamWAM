"""Independent graph dispatch for the original DiT pre/post methods.

Compose with equally optimized Dense or temporal transformer controls. Each
original pre/post method still runs every denoising step, with refreshed inputs
and cloned outputs. Samplers, selection, world routing and schedulers are intact.
No production policy option is enabled by this experimental benchmark factor.
"""

from __future__ import annotations

from contextlib import ExitStack

from .visual_cache_graphs import TensorTreeReplay


class DiTBoundaryGraphs:
    names = ("video_pre", "action_pre", "video_post", "action_post")

    def __init__(self, model, *, enabled=True, warmup=3):
        if model.training or model.config.setting != "joint":
            raise ValueError("DiT boundary replay requires Joint inference")
        self.model, self.enabled, self.warmup = model, enabled, warmup
        self.graphs = {}
        self._originals = []
        self._active = False
        self.last_stats = {}

    def _call(self, name, function, request):
        if not self._active:
            raise RuntimeError("DiT boundary replay requires a sampling request")
        if name not in self.graphs:
            self.graphs[name] = TensorTreeReplay(
                function, request, device=next(self.model.parameters()).device,
                enabled=self.enabled, warmup=self.warmup)
        replay = self.graphs[name]
        result = replay(request)
        self._stats[f"dit_{name}_calls"] += 1
        self._stats[f"dit_{name}_graph_replays"] += int(replay.graph is not None)
        return result

    def __enter__(self):
        if self._originals or getattr(self.model, "_dit_boundary_graphs", None):
            raise RuntimeError("another DiT boundary wrapper is already installed")
        video, action = self.model.video_expert, self.model.action_expert
        video_pre, action_pre = video.pre_dit, action.pre_dit
        video_post, action_post = video.post_dit, action.post_dit
        sample = self.model.sample_action

        def pre_video(**kwargs):
            return self._call("video_pre", video_pre, kwargs)

        def pre_action(**kwargs):
            return self._call("action_pre", action_pre, kwargs)

        def post_video(tokens, state):
            # These are the only state fields read by the original video head.
            return self._call("video_post", video_post, dict(tokens=tokens, state=dict(
                time_embedding=state["time_embedding"], grid_size=state["grid_size"])))

        def post_action(tokens, state=None):
            # ActionDiT.post_dit explicitly discards the optional state.
            return self._call("action_post", action_post, dict(tokens=tokens))

        def sample_action(*args, **kwargs):
            if self._active:
                raise RuntimeError("concurrent/reentrant boundary replay is unsupported")
            self._active = True
            self._stats = {f"dit_{name}_{field}": 0 for name in self.names
                           for field in ("calls", "graph_replays")}
            try:
                return sample(*args, **kwargs)
            finally:
                self.last_stats = dict(self._stats)
                self._active = False

        self._originals = [(video, "pre_dit", video_pre), (action, "pre_dit", action_pre),
                           (video, "post_dit", video_post), (action, "post_dit", action_post),
                           (self.model, "sample_action", sample)]
        video.pre_dit, action.pre_dit = pre_video, pre_action
        video.post_dit, action.post_dit = post_video, post_action
        self.model.sample_action = sample_action
        self.model._dit_boundary_graphs = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for owner, name, original in reversed(self._originals):
            setattr(owner, name, original)
        self._originals.clear()
        self._active = False
        del self.model._dit_boundary_graphs

    def graph_stats(self):
        return {f"dit_{name}": dict(captured=replay.graph is not None,
                                   capture_seconds=replay.capture_seconds, replays=replay.replays)
                for name, replay in self.graphs.items()}

    def close_graphs(self):
        if self._active:
            raise RuntimeError("cannot release boundary graphs during a request")
        self.graphs.clear()


class BoundaryGraphControl:
    """Benchmark composition with the existing transformer control unchanged."""

    def __init__(self, transformer, *, enabled=True, warmup=3):
        self.transformer = transformer
        self.boundaries = DiTBoundaryGraphs(transformer.model, enabled=enabled, warmup=warmup)
        self._stack = None

    def __getattr__(self, name):
        return getattr(self.transformer, name)

    @property
    def last_stats(self):
        return {**self.transformer.last_stats, **self.boundaries.last_stats}

    def __enter__(self):
        if self._stack is not None:
            raise RuntimeError("boundary control is already installed")
        with ExitStack() as stack:
            stack.enter_context(self.transformer)
            stack.enter_context(self.boundaries)
            self._stack = stack.pop_all()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        stack, self._stack = self._stack, None
        return stack.__exit__(exc_type, exc_value, traceback)

    def graph_stats(self):
        return {**self.transformer.graph_stats(), **self.boundaries.graph_stats()}

    def close_graphs(self):
        self.boundaries.close_graphs()
        self.transformer.close_graphs()
