"""Bounded, replayable native Dense traces with exception-safe request isolation."""

from dataclasses import asdict
from functools import wraps

import torch

from .geometry import TokenGrid
from .signals import joint_probabilities, token_signals, value_denoising_drift


def indices(values, upper, name):
    values = tuple(values)
    if not values or len(set(values)) != len(values) or any(type(x) is not int or not 0 <= x < upper for x in values):
        raise ValueError(f"{name} must be unique indices in [0, {upper})")
    return values


def raw_numpy(tensor):
    """Float32 stores bf16/fp16 projection values exactly; no additional loss."""
    tensor = tensor.detach().cpu()
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.float()
    return tensor.numpy().copy()


class DenseProfile:
    """Invoke sink(kind, metadata, arrays) once per selected layer/step.

    Does not replace native attention output, model parameters or input tensors.
    Save one profile per sample_action call; repeated calls get isolated state
    and a monotonically increasing request number. Sink failures propagate.
    """

    def __init__(self, model, *, steps, layers, heads, sink, cameras=2,
                 max_records=40, max_bytes=256 * 1024**2):
        if model.training or model.config.setting != "joint":
            raise ValueError("profiling requires native Joint inference")
        self.steps = indices(steps, 1000, "steps")
        self.layers = indices(layers, model.mot.num_layers, "layers")
        self.heads = indices(heads, model.mot.num_heads, "heads")
        if type(max_records) is not int or max_records < len(self.steps) * len(self.layers):
            raise ValueError("record cap is below requested sampling coverage")
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("byte cap must be positive")
        self.model, self.sink, self.cameras = model, sink, cameras
        self.max_records, self.max_bytes = max_records, max_bytes
        self.originals = []
        self.request = 0
        self.active = False
        self.stats = {}
        self.previous = {}
        self.raw_action = self.last_video = None
        self.grid = None

    def emit(self, kind, metadata, arrays):
        arrays = {key: raw_numpy(value) if isinstance(value, torch.Tensor) else value
                  for key, value in arrays.items()}
        byte_count = sum(value.nbytes for value in arrays.values())
        if self.stats["raw_bytes"] + byte_count > self.max_bytes:
            raise RuntimeError("profile raw-byte budget exceeded; partial evidence retained by sink")
        self.sink(kind, dict(request=self.request, **metadata), arrays)
        self.stats["raw_bytes"] += byte_count

    def __enter__(self):
        if self.originals or getattr(self.model, "_visual_ffn_context_cache", None):
            raise RuntimeError("profile requires unwrapped native Dense")
        self.model._visual_ffn_context_cache = self
        mot = self.model.mot
        sample, forward, attention = self.model.sample_action, mot.forward, mot._joint_self_attention
        video_step = self.model.video_scheduler.step

        @wraps(sample)
        def observe_sample(*args, **kwargs):
            if self.active:
                raise RuntimeError("concurrent/reentrant profiling is unsupported")
            self.previous.clear()
            self.grid = self.raw_action = self.last_video = None
            count = kwargs.get("num_steps", 10)
            if any(step >= count for step in self.steps):
                raise ValueError("profile steps exceed the unchanged sampler")
            if kwargs.get("sparse") is not None and kwargs["sparse"].enabled:
                raise ValueError("profile must follow the full native Dense trajectory")
            self.request += 1
            self.stats = dict(status="RUNNING", records=0, raw_bytes=0, video_steps=0,
                              observed_steps=[], observed_layer_steps=[])
            self.active = True
            try:
                result = sample(*args, **kwargs)
                wanted = {(s, l) for s in self.steps for l in self.layers}
                actual = set(map(tuple, self.stats["observed_layer_steps"]))
                if actual != wanted or self.stats["records"] != len(wanted) or self.stats["video_steps"] != count:
                    raise RuntimeError("native sampling coverage differs from the declared profile")
                self.raw_action = result.detach().clone()
                # Native sampler restores the observed frame in this same video tensor.
                self.last_video = self.last_video.detach().clone()
                self.emit("final", dict(grid=asdict(self.grid)),
                          dict(raw_action=result, video_latents=self.last_video))
                self.stats["status"] = "COMPLETE"
                return result
            except BaseException:
                self.stats["status"] = "ERROR"
                self.raw_action = self.last_video = None
                raise
            finally:
                self.active = False
                self.previous.clear()

        @wraps(forward)
        def observe_forward(**kwargs):
            if not self.active:
                raise RuntimeError("profile must enter through sample_action")
            grid = TokenGrid(*map(int, kwargs["video_state"]["grid_size"]), cameras=self.cameras)
            if self.grid is not None and self.grid != grid:
                raise ValueError("grid changed inside a request")
            self.grid = grid
            if kwargs["step_index"] in self.steps:
                arrays = dict(coordinates=grid.coordinates())
                for name in ("video", "action"):
                    freqs = kwargs[name + "_state"]["freqs"]
                    if isinstance(freqs, tuple):
                        arrays[name + "_rope_cos"], arrays[name + "_rope_sin"] = freqs
                    else:
                        arrays[name + "_rope_complex"] = freqs
                self.emit("step", dict(step=kwargs["step_index"], grid=asdict(grid)), arrays)
                self.stats["observed_steps"].append(kwargs["step_index"])
            return forward(**kwargs)

        @wraps(attention)
        def observe_attention(**kwargs):
            if kwargs["sparse"] is not None and kwargs["sparse"].enabled:
                raise ValueError("sparse attention cannot be profiled as Dense")
            result = attention(**kwargs)
            step, layer = kwargs["step_index"], kwargs["layer_index"]
            if step not in self.steps or layer not in self.layers:
                return result
            if self.stats["records"] >= self.max_records:
                raise RuntimeError("profile record cap exceeded")
            if kwargs["video_length"] != self.grid.length or kwargs["tokens_per_frame"] != self.grid.frame_size:
                raise ValueError("actual attention layout differs from the recorded spatial grid")
            tensors = []
            for i in range(3):
                combined = torch.cat((kwargs["video_io"][i], kwargs["action_io"][i]), dim=1)
                if combined.shape[0] != 1:
                    raise ValueError("bounded raw profile requires batch one")
                heads = combined[0].reshape(combined.shape[1], mot.num_heads, -1).transpose(0, 1)
                tensors.append(heads[list(self.heads)].detach().float().cpu())
            query, key, value = tensors
            mask = kwargs["attention_mask"].detach().cpu()
            probabilities = joint_probabilities(query, key, mask)
            arrays = dict(query=query, key=key, value=value, mask=mask,
                          av_probabilities=probabilities[:, self.grid.length:],
                          vv_probabilities=probabilities[:, :self.grid.length])
            arrays.update({"score_" + k: v for k, v in token_signals(probabilities, value, self.grid).items()})
            before = self.previous.get(layer)
            if before is not None:
                arrays["value_denoising_drift"] = value_denoising_drift(value, before[1], self.grid.length)
            metadata = dict(step=step, layer=layer, heads=list(self.heads), grid=asdict(self.grid),
                video_length=self.grid.length, action_length=query.shape[1] - self.grid.length,
                original_dtype=str(kwargs["video_io"][0].dtype),
                previous_sampled_step=None if before is None else before[0],
                adjacent_denoising_step=before is not None and before[0] == step - 1,
                qk_position="native_post_rope", value_position="native_unrotated",
                denominator="all_native_visual_and_action_keys_with_original_mask")
            self.emit("attention", metadata, arrays)
            self.previous[layer] = (step, value)
            self.stats["records"] += 1
            self.stats["observed_layer_steps"].append([step, layer])
            return result

        @wraps(video_step)
        def observe_video(*args, **kwargs):
            result = video_step(*args, **kwargs)
            step = self.stats["video_steps"]
            self.last_video = result
            if step in self.steps:
                self.emit("latent", dict(step=step, observed_frame_restored=False), dict(video_latents=result))
            self.stats["video_steps"] += 1
            return result

        self.originals = [(self.model, "sample_action", sample), (mot, "forward", forward),
                          (mot, "_joint_self_attention", attention),
                          (self.model.video_scheduler, "step", video_step)]
        self.model.sample_action, mot.forward = observe_sample, observe_forward
        mot._joint_self_attention, self.model.video_scheduler.step = observe_attention, observe_video
        return self

    def __exit__(self, *args):
        for owner, name, value in reversed(self.originals):
            setattr(owner, name, value)
        self.originals.clear()
        self.previous.clear()
        self.raw_action = self.last_video = None
        if getattr(self.model, "_visual_ffn_context_cache", None) is self:
            del self.model._visual_ffn_context_cache
