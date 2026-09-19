"""Coarse temporal visual-token reuse, with every action denoising step retained.

Dense refresh steps capture post-RoPE visual K/V at every Joint layer and the
final visual hidden states (including world residuals). Between refreshes, the
unchanged action expert attends to those cached K/V through the model's existing
action-with-video-cache path. Current video pre/post-DiT and both schedulers still
run. This is a separately measured temporal reuse ablation, not token selection
by action relevance, and it must not be reported as changing denoising steps.
"""

from __future__ import annotations

from functools import wraps
from collections.abc import Mapping
import math

from .ffn_context_cache import VisualFFNContextCache


def visual_cache_options(payload):
    """Strict adapter/policy configuration; omission retains native Dense."""
    if payload is None:
        return None
    allowed = {"refresh_every", "token_keep_ratio", "action_guidance_weight", "graph_dispatch"}
    if not isinstance(payload, Mapping) or "refresh_every" not in payload or set(payload) - allowed:
        raise ValueError("visual_cache requires refresh_every; optional keys are token_keep_ratio, action_guidance_weight and graph_dispatch")
    interval = payload["refresh_every"]
    if isinstance(interval, bool) or not isinstance(interval, int) or interval < 1:
        raise ValueError("visual_cache.refresh_every must be a positive integer")
    options = {"refresh_every": interval}
    if "token_keep_ratio" not in payload:
        if "action_guidance_weight" in payload or "graph_dispatch" in payload:
            raise ValueError("visual_cache.action_guidance_weight and graph_dispatch require token_keep_ratio")
        return options  # Preserve the existing temporal-only run identity exactly.
    ratio = payload["token_keep_ratio"]
    weight = payload.get("action_guidance_weight", 0.0)
    for name, value in (("token_keep_ratio", ratio), ("action_guidance_weight", weight)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"visual_cache.{name} must be a finite number")
    if not 0 <= ratio <= 1:
        raise ValueError("visual_cache.token_keep_ratio must be in [0, 1]")
    if weight < 0:
        raise ValueError("visual_cache.action_guidance_weight must be non-negative")
    options.update(token_keep_ratio=float(ratio), action_guidance_weight=float(weight))
    if "graph_dispatch" in payload:
        mode = payload["graph_dispatch"]
        if not isinstance(mode, str) or mode not in {"dense_action", "all_transformers"}:
            raise ValueError("visual_cache.graph_dispatch must be dense_action or all_transformers")
        options["graph_dispatch"] = mode
    return options


class VisualStepCache(VisualFFNContextCache):
    def __init__(self, model, *, refresh_every: int = 5):
        super().__init__(model, keep_ratio=1.0)
        if isinstance(refresh_every, bool) or not isinstance(refresh_every, int) or refresh_every < 1:
            raise ValueError("refresh_every must be a positive integer")
        self.refresh_every = refresh_every
        self.video_kv = {}
        self.video_output = None
        self._capture = False
        self._video_layers = {id(block): index for index, block in enumerate(model.video_expert.blocks)}

    def _begin(self):
        super()._begin()
        self.video_kv.clear()
        self.video_output = None
        self._stats = dict(dense_video_steps=0, reused_video_steps=0,
                           video_layer_updates=0, action_layer_updates=0)

    def _end(self):
        super()._end()
        self.video_kv.clear()
        self.video_output = None
        self._capture = False

    def _refresh(self, forward, **kwargs):
        self._capture = True
        try:
            result = forward(**kwargs)
        finally:
            self._capture = False
        self.video_output = result["video"]
        self._stats["dense_video_steps"] += 1
        self._stats["video_layer_updates"] += self.model.mot.num_layers
        if len(self.video_kv) != self.model.mot.num_layers:
            raise RuntimeError("dense refresh did not capture all visual layers")
        return result

    def _reuse(self, *, video_state, action_state):
        if self.video_output is None:
            raise RuntimeError("visual cache has no current-request dense anchor")
        if self.video_output.shape != video_state["tokens"].shape:
            raise ValueError("visual layout changed within a request")
        mot = self.model.mot
        action = mot.forward_action_with_video_cache(
            action_state=action_state,
            video_kv_cache=[self.video_kv[layer] for layer in range(mot.num_layers)],
            video_length=self.video_output.shape[1],
            video_tokens_per_frame=video_state["tokens_per_frame"],
        )
        self._stats["reused_video_steps"] += 1
        return {"video": self.video_output, "action": action}

    def __enter__(self):
        if self._originals or getattr(self.model, "_visual_ffn_context_cache", None):
            raise RuntimeError("another visual cache is already installed")
        self.model._visual_ffn_context_cache = self
        mot = self.model.mot
        attention_input = mot._attention_input

        @wraps(attention_input)
        def capture_input(block, *args, **kwargs):
            result = attention_input(block, *args, **kwargs)
            index = self._video_layers.get(id(block))
            if self._capture and index is not None:
                self.video_kv[index] = {"k": result[1], "v": result[2]}
            return result

        forward = mot.forward

        @wraps(forward)
        def joint_forward(*, video_state, action_state, residual_injection,
                          sparse=None, step_index=0, num_steps=1):
            if not self._active:
                raise RuntimeError("visual reuse must run inside sample_action")
            if sparse is not None and sparse.enabled:
                raise ValueError("visual-step factor must be measured without sparse attention")
            self._stats["action_layer_updates"] += mot.num_layers
            if step_index % self.refresh_every == 0:
                return self._refresh(forward, video_state=video_state, action_state=action_state,
                                     residual_injection=residual_injection, sparse=sparse,
                                     step_index=step_index, num_steps=num_steps)
            return self._reuse(video_state=video_state, action_state=action_state)

        sample = self.model.sample_action

        @wraps(sample)
        def sample_action(*args, **kwargs):
            self._begin()
            try:
                return sample(*args, **kwargs)
            finally:
                self._end()

        self._originals.extend([(mot, "_attention_input", attention_input),
                                (mot, "forward", forward),
                                (self.model, "sample_action", sample)])
        mot._attention_input = capture_input
        mot.forward = joint_forward
        self.model.sample_action = sample_action
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        self.video_kv.clear()
        self.video_output = None
        self._capture = False
