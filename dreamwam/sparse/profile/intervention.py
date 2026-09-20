"""Dense diagnostic interventions separating direct AV reads from VV context.

All projection/FFN work still executes. Recompute is a counterfactual choice of
current versus previous-step K/V, NOT an accelerated sparse implementation.
"""

from functools import wraps

import torch


def intervention_mask(mask, indices, video_length, scope):
    if mask.dtype != torch.bool or mask.ndim != 2 or mask.shape[0] != mask.shape[1]:
        raise ValueError("require original square joint boolean mask")
    if scope not in ("AV", "VV", "joint"):
        raise ValueError("scope must distinguish AV, VV or joint")
    result = mask.clone()
    start, stop = ((video_length, mask.shape[0]) if scope == "AV" else
                   (0, video_length) if scope == "VV" else (0, mask.shape[0]))
    result[start:stop, indices] = False
    if not result.any(dim=-1).all():
        raise ValueError("intervention would remove every visible key")
    return result


class KeyIntervention:
    def __init__(self, model, targets, *, operation, scope):
        if model.training or model.config.setting != "joint":
            raise ValueError("interventions require native Joint inference")
        if operation not in ("delete", "replace_value_zero", "recompute") or scope not in ("AV", "VV", "joint"):
            raise ValueError("unsupported intervention operation/scope")
        self.targets = {}
        for (step, layer), values in targets.items():
            if type(step) is not int or step < 0 or type(layer) is not int or not 0 <= layer < model.mot.num_layers:
                raise ValueError("invalid target step/layer")
            if operation == "recompute" and step == 0:
                raise ValueError("recompute intervention requires a previous denoising step")
            values = tuple(values)
            if len(set(values)) != len(values) or any(type(v) is not int or v < 0 for v in values):
                raise ValueError("target indices must be unique nonnegative integers")
            self.targets[step, layer] = values
        self.model, self.operation, self.scope = model, operation, scope
        self.originals = []
        self.previous = {}
        self.active = False
        self.records = []
        self.raw_action = self.last_video = None

    def __enter__(self):
        if self.originals or getattr(self.model, "_visual_ffn_context_cache", None):
            raise RuntimeError("intervention requires an isolated native model")
        self.model._visual_ffn_context_cache = self
        mot = self.model.mot
        attention, sample, video_step = mot._joint_self_attention, self.model.sample_action, self.model.video_scheduler.step

        @wraps(attention)
        def intervene(**kwargs):
            if kwargs["sparse"] is not None and kwargs["sparse"].enabled:
                raise ValueError("cannot combine diagnostic and accelerated sparsity")
            step, layer, nv = kwargs["step_index"], kwargs["layer_index"], kwargs["video_length"]
            video_io = kwargs["video_io"]
            target = self.targets.get((step, layer))
            if target is not None and any(i >= nv for i in target):
                raise ValueError("intervention may only target original visual indices")
            if target is None:
                result = attention(**kwargs)
            else:
                ids = torch.tensor(target, dtype=torch.long, device=video_io[0].device)
                modified = dict(kwargs)
                if self.operation == "delete":
                    modified["attention_mask"] = intervention_mask(kwargs["attention_mask"], ids, nv, self.scope)
                    result = attention(**modified)
                else:
                    vio = list(video_io)
                    if self.operation == "replace_value_zero":
                        vio[2] = vio[2].index_fill(1, ids, 0)
                    else:
                        before = self.previous.get(layer)
                        if before is None or before[0] != step - 1:
                            raise RuntimeError("no adjacent previous-step K/V for recompute diagnostic")
                        vio[1] = before[1].index_copy(1, ids, vio[1].index_select(1, ids))
                        vio[2] = before[2].index_copy(1, ids, vio[2].index_select(1, ids))
                    modified["video_io"] = tuple(vio)
                    result = attention(**modified)
                    if self.scope != "joint":
                        original = attention(**kwargs)
                        result = (torch.cat((original[:, :nv], result[:, nv:]), dim=1)
                                  if self.scope == "AV" else
                                  torch.cat((result[:, :nv], original[:, nv:]), dim=1))
                self.records.append(dict(step=step, layer=layer, scope=self.scope, operation=self.operation,
                    indices=list(target), count=len(target), video_length=nv,
                    action_length=kwargs["action_io"][0].shape[1], heads=mot.num_heads,
                    full_projection_work_retained=True))
            if self.operation == "recompute":
                self.previous[layer] = (step, video_io[1].detach().clone(), video_io[2].detach().clone())
            return result

        @wraps(video_step)
        def capture_video(*args, **kwargs):
            result = video_step(*args, **kwargs)
            self.last_video = result
            return result

        @wraps(sample)
        def run(*args, **kwargs):
            if self.active:
                raise RuntimeError("concurrent/reentrant diagnostic is unsupported")
            self.previous.clear()
            self.records.clear()
            self.raw_action = self.last_video = None
            if any(step >= kwargs.get("num_steps", 10) for step, _ in self.targets):
                raise ValueError("intervention outside sampler schedule")
            self.active = True
            try:
                result = sample(*args, **kwargs)
                if len(self.records) != len(self.targets):
                    raise RuntimeError("intervention coverage incomplete")
                self.raw_action = result.detach().clone()
                self.last_video = self.last_video.detach().clone()
                return result
            except BaseException:
                self.raw_action = self.last_video = None
                raise
            finally:
                self.previous.clear()
                self.active = False

        self.originals = [(mot, "_joint_self_attention", attention), (self.model, "sample_action", sample),
                          (self.model.video_scheduler, "step", video_step)]
        mot._joint_self_attention, self.model.sample_action = intervene, run
        self.model.video_scheduler.step = capture_video
        return self

    def __exit__(self, *args):
        for owner, name, original in reversed(self.originals):
            setattr(owner, name, original)
        self.originals.clear()
        self.previous.clear()
        self.raw_action = self.last_video = None
        if getattr(self.model, "_visual_ffn_context_cache", None) is self:
            del self.model._visual_ffn_context_cache
