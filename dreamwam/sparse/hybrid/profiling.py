"""Offline-only dense dependency audit and controlled key-removal interventions.

Never used to select online routes. This deliberately retains full tensor sizes;
its latency is NOT a sparse speed measurement. No benchmark labels are accessed.
"""

from dataclasses import replace
import math
from types import SimpleNamespace

import torch

from .config import HybridConfig, number
from .routing import decision_scores
from .schedule import Schedule


def balanced_indices(score, length, frame_size, ratio, *, bottom=False, device=None):
    number(ratio, "removal ratio", 0, 1, open_low=True)
    if ratio == 1 or length % frame_size:
        raise ValueError("removal must leave visible keys in every complete frame")
    count = math.ceil(length * ratio)
    frames = length // frame_size
    parts = []
    for frame in range(frames):
        size = count // frames + int(frame < count % frames)
        if size >= frame_size:
            raise ValueError("removal would empty a frame")
        if score is None:
            local = torch.arange(size, device=device) * frame_size // max(size, 1)
        else:
            local = score[frame * frame_size:(frame + 1) * frame_size].argsort(
                descending=not bottom, stable=True)[:size]
        parts.append(local + frame * frame_size)
    return torch.cat(parts).sort().values


class DependencyAudit:
    def __init__(self, model, *, remove_step=None, method="action", ratio=0.1,
                 collect=False, context_weight=1.0, scope="all", group_index=None, group_count=7):
        if model.training or model.config.setting != "joint":
            raise ValueError("dependency audit requires Joint inference")
        if method not in ("uniform", "action", "bottom_action", "visual_context", "action_context", "group"):
            raise ValueError("unsupported intervention selector")
        if scope not in ("all", "future"):
            raise ValueError("scope must be all or future")
        if method == "group" and (type(group_count) is not int or group_count < 2 or
                type(group_index) is not int or not 0 <= group_index < group_count):
            raise ValueError("require a valid group index and group count >= 2")
        number(ratio, "removal ratio", 0, 1, open_low=True)
        self.model, self.remove_step, self.method = model, remove_step, method
        self.ratio, self.collect, self.context_weight = ratio, collect, context_weight
        self.scope, self.group_index, self.group_count = scope, group_index, group_count
        self.records, self.removed, self.future_latents = [], None, None
        self.modified_layers = 0
        self.originals = []

    def _indices(self, score, length, frame_size, *, bottom=False, device=None):
        offset = frame_size if self.scope == "future" else 0
        if self.method == "group" and score is None:
            if self.group_count > frame_size:
                raise ValueError("group count exceeds tokens per frame")
            start = self.group_index * frame_size // self.group_count
            end = (self.group_index + 1) * frame_size // self.group_count
            return torch.cat([torch.arange(start, end, device=device) + frame
                              for frame in range(offset, length, frame_size)])
        return balanced_indices(None if score is None else score[offset:], length - offset,
                                frame_size, self.ratio, bottom=bottom, device=device) + offset

    def __enter__(self):
        if self.originals or getattr(self.model, "_visual_ffn_context_cache", None):
            raise RuntimeError("audit is isolated from acceleration wrappers")
        mot = self.model.mot
        forward, joint, video_step = mot.forward, mot._joint_self_attention, self.model.video_scheduler.step

        def observe(**kwargs):
            step = kwargs["step_index"]
            video, action = kwargs["video_state"], kwargs["action_state"]
            length, frame_size = video["tokens"].shape[1], video["tokens_per_frame"]
            config = HybridConfig(Schedule(1, ("dense",)), selection="action",
                                  context_weight=self.context_weight)
            mask = mot.build_attention_mask(video_length=length,
                action_length=action["tokens"].shape[1], video_tokens_per_frame=frame_size,
                device=video["tokens"].device)
            context = SimpleNamespace(full_mask=mask)
            methods = ("action", "visual_context", "action_context") if self.collect else ()
            selected_method = "action" if self.method == "bottom_action" else self.method
            if step == self.remove_step and selected_method not in ("uniform", "group") and selected_method not in methods:
                methods += (selected_method,)
            values = {method: decision_scores(replace(config, selection=method), self.model,
                        video, action, context, anchor=True).read for method in methods}
            if self.collect:
                self.records.append(dict(step=step, tokens_per_frame=frame_size, length=length,
                    scores={method: value.detach().cpu().tolist() for method, value in values.items()},
                    top={method: self._indices(value, length, frame_size).cpu().tolist()
                         for method, value in values.items()}))
            if step == self.remove_step:
                self.removed = self._indices(values.get(selected_method), length, frame_size,
                    bottom=self.method == "bottom_action", device=video["tokens"].device)
            return forward(**kwargs)

        def intervene(**kwargs):
            if kwargs["step_index"] == self.remove_step:
                mask = kwargs["attention_mask"].clone()
                mask[:, self.removed] = False
                if not mask.any(dim=-1).all():
                    raise ValueError("intervention leaves an attention row without visible keys")
                kwargs["attention_mask"] = mask
                self.modified_layers += 1
            return joint(**kwargs)

        def capture(*args, **kwargs):
            result = video_step(*args, **kwargs)
            self.future_latents = result[:, :, 1:].detach().clone()
            return result

        self.originals = [(mot, "forward", forward), (mot, "_joint_self_attention", joint),
                          (self.model.video_scheduler, "step", video_step)]
        mot.forward, mot._joint_self_attention, self.model.video_scheduler.step = observe, intervene, capture
        return self

    def __exit__(self, *args):
        for owner, name, value in reversed(self.originals):
            setattr(owner, name, value)
        self.originals.clear()


def route_stability(records):
    """Jaccard on fixed-budget supports, reported at each denoising transition."""
    rows = []
    for before, after in zip(records, records[1:]):
        for method in before["top"]:
            a, b = set(before["top"][method]), set(after["top"][method])
            rows.append(dict(before=before["step"], after=after["step"], method=method,
                             jaccard=len(a & b) / len(a | b)))
    return rows
