"""Fresh compact visual transformers at EVERY denoising step, including step 0.

No visual K/V, hidden state or prediction is retained from a previous step.
Each step selects current-input rows, computes all transformer layers on those
rows, and scatters them into the CURRENT pre-DiT input. Unselected rows bypass
the transformer; the original full-grid prediction head and schedulers still
run. Action rows are all recomputed and attend only to the selected current
visual keys plus all action keys. This changes AV support, not just VV pairs.

Frame quotas preserve observed-frame support within ceil(N * keep_ratio).
Action selection probes current first-layer video K and action Q/K, with joint
A->[V,A] normalization. Its full-video K projection is real additional work.
CUDA graphs only replay computation: all inputs are overwritten on every call.
"""

from collections.abc import Mapping
import math

import torch

from dreamwam.layers import apply_rope, modulate, scaled_dot_product_attention
from .action_guided_neuron_cache import joint_video_mass
from .visual_cache_graphs import GraphedVisualTokenCache, TensorTreeReplay


def fresh_visual_options(payload):
    if payload is None:
        return None
    allowed = {"keep_ratio", "selection", "graph_dispatch"}
    if not isinstance(payload, Mapping) or "keep_ratio" not in payload or set(payload) - allowed:
        raise ValueError("fresh_visual_tokens requires keep_ratio; optional selection and graph_dispatch")
    ratio = payload["keep_ratio"]
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not math.isfinite(ratio) or not 0 < ratio <= 1:
        raise ValueError("fresh_visual_tokens.keep_ratio must be finite and in (0, 1]")
    selection = payload.get("selection", "action")
    if selection not in ("action", "uniform"):
        raise ValueError("fresh_visual_tokens.selection must be action or uniform")
    result = dict(keep_ratio=float(ratio), selection=selection)
    if "graph_dispatch" in payload:
        if payload["graph_dispatch"] != "all_transformers":
            raise ValueError("fresh_visual_tokens.graph_dispatch must be all_transformers")
        result["graph_dispatch"] = "all_transformers"
    return result


class FreshVisualTokenSparsity:
    def __init__(self, model, *, keep_ratio, selection="action", graph_enabled=False,
                 buffered=False, graph_warmup=3):
        options = fresh_visual_options(dict(keep_ratio=keep_ratio, selection=selection))
        if model.training or model.config.setting != "joint":
            raise ValueError("fresh visual sparsity requires Joint inference")
        self.model = model
        self.keep_ratio = options["keep_ratio"]
        self.selection = options["selection"]
        self.graph_enabled = graph_enabled
        self.buffered = buffered or graph_enabled
        self.graph_warmup = graph_warmup
        self.replay = None
        self._originals = []
        self._active = False
        self._indices = []
        self.last_stats = {}

    def _select(self, video_state, action_state, full_mask):
        mot = self.model.mot
        current = video_state["tokens"]
        length = current.shape[1]
        frame_size = video_state["tokens_per_frame"]
        if length % frame_size:
            raise ValueError("fresh visual selection requires complete frame grids")
        frames = length // frame_size
        keep = math.ceil(length * self.keep_ratio)
        if keep < frames:
            raise ValueError("token budget must retain at least one token per frame")
        first_action_io = None
        scores = None
        if self.selection == "action":
            block = mot.video_expert.blocks[0]
            shift, scale, *_ = mot._split_modulation(block, video_state["time_modulation"])
            current_key = apply_rope(
                block.self_attn.norm_k(block.self_attn.k(modulate(block.norm1(current), shift, scale))),
                video_state["freqs"], mot.num_heads,
            )
            first_action_io = mot._attention_input(
                mot.action_expert.blocks[0], action_state["tokens"],
                action_state["freqs"], action_state["time_modulation"],
            )
            scores = joint_video_mass(first_action_io[0], current_key, first_action_io[1],
                                     num_heads=mot.num_heads, action_mask=full_mask[length:]).mean(dim=0)
        indices = []
        for frame in range(frames):
            count = keep // frames + int(frame < keep % frames)
            if scores is None:
                local = torch.arange(count, device=current.device) * frame_size // count
            else:
                local = scores[frame * frame_size:(frame + 1) * frame_size].argsort(
                    descending=True, stable=True)[:count]
            indices.append(local + frame * frame_size)
        return torch.cat(indices).sort().values, first_action_io

    def _compute(self, video_state, action_state):
        mot = self.model.mot
        router = self.model.world_residual
        router.reset_runtime()
        current = video_state["tokens"]
        length = current.shape[1]
        if self.keep_ratio == 1:
            result = self._dense_forward(video_state=video_state, action_state=action_state,
                                         residual_injection=router, sparse=None)
            index = torch.arange(length, device=current.device)
        else:
            action = action_state["tokens"]
            full_mask = mot.build_attention_mask(video_length=length, action_length=action.shape[1],
                video_tokens_per_frame=video_state["tokens_per_frame"], device=current.device)
            index, first_action_io = self._select(video_state, action_state, full_mask)
            keep = index.numel()
            joint_index = torch.cat((index, torch.arange(action.shape[1], device=current.device) + length))
            mask = full_mask.index_select(0, joint_index).index_select(1, joint_index)
            video = current.index_select(1, index)
            freqs = video_state["freqs"]
            freqs = (tuple(x.index_select(0, index) for x in freqs) if isinstance(freqs, tuple)
                     else freqs.index_select(0, index))
            modulation = video_state["time_modulation"]
            if modulation.ndim == 4:
                modulation = modulation.index_select(1, index)
            context_mask = video_state["context_mask"]
            if context_mask.ndim == 3:
                context_mask = context_mask.index_select(1, index)
            for layer in range(mot.num_layers):
                vb, ab = mot.video_expert.blocks[layer], mot.action_expert.blocks[layer]
                vio = mot._attention_input(vb, video, freqs, modulation)
                aio = (first_action_io if layer == 0 and first_action_io is not None else
                       mot._attention_input(ab, action, action_state["freqs"], action_state["time_modulation"]))
                mixed = scaled_dot_product_attention(
                    torch.cat((vio[0], aio[0]), dim=1), torch.cat((vio[1], aio[1]), dim=1),
                    torch.cat((vio[2], aio[2]), dim=1), mot.num_heads, mask,
                )
                video = mot._post_attention(vb, vio[3], mixed[:, :keep], *vio[4:],
                                            video_state["context"], context_mask)
                action = mot._post_attention(ab, aio[3], mixed[:, keep:], *aio[4:],
                                             action_state["context"], action_state["context_mask"])
                video = router(layer, video, video_state["context"], context_mask)
            # Identity bypass from this step's input, NEVER a previous step's output.
            result = dict(video=current.index_copy(1, index, video), action=action)
        return dict(result=result, indices=index,
                    gates={k: list(v) for k, v in router._runtime_gates.items()},
                    previews={k: list(v) for k, v in router._runtime_local_preview_tokens.items()})

    def __enter__(self):
        if self._originals or getattr(self.model, "_visual_ffn_context_cache", None):
            raise RuntimeError("another inference wrapper is already installed")
        # Shared exclusion marker used by existing wrappers; it holds no cached tensors.
        self.model._visual_ffn_context_cache = self
        mot = self.model.mot
        self._dense_forward = mot.forward

        def forward(*, video_state, action_state, residual_injection, sparse=None,
                    step_index=0, num_steps=1):
            if not self._active or step_index != self._stats["denoising_steps"]:
                raise RuntimeError("fresh visual sparsity requires sequential sample_action steps")
            if residual_injection is not self.model.world_residual or (sparse is not None and sparse.enabled):
                raise ValueError("measure fresh visual sparsity without another attention/router factor")
            request = dict(video_state=GraphedVisualTokenCache._state(video_state, video=True),
                           action_state=GraphedVisualTokenCache._state(action_state))
            if self.buffered:
                if self.replay is None:
                    self.replay = TensorTreeReplay(self._compute, request,
                        device=next(self.model.parameters()).device,
                        enabled=self.graph_enabled, warmup=self.graph_warmup)
                output = self.replay(request)
            else:
                output = self._compute(**request)
            router = self.model.world_residual
            router.reset_runtime()
            for k, v in output["gates"].items():
                router._runtime_gates[k].extend(v)
            for k, v in output["previews"].items():
                router._runtime_local_preview_tokens[k].extend(v)
            batch, length = video_state["tokens"].shape[:2]
            keep = output["indices"].numel()
            self._indices.append(output["indices"])
            self._stats["denoising_steps"] += 1
            self._stats["action_layer_updates"] += mot.num_layers
            self._stats["computed_video_token_layers"] += batch * keep * mot.num_layers
            self._stats["total_video_token_layers"] += batch * length * mot.num_layers
            self._stats["selection_builds"] += int(self.keep_ratio < 1)
            self._stats["selection_video_key_rows"] += batch * length * int(self.keep_ratio < 1 and self.selection == "action")
            self._stats["graph_replays"] += int(self.replay is not None and self.replay.graph is not None)
            return output["result"]

        sample = self.model.sample_action

        def sample_action(*args, **kwargs):
            if self._active:
                raise RuntimeError("concurrent/reentrant sampling is unsupported")
            self._active = True
            self._indices = []
            self._stats = dict(denoising_steps=0, action_layer_updates=0,
                computed_video_token_layers=0, total_video_token_layers=0, selection_builds=0,
                selection_video_key_rows=0, reused_visual_steps=0, graph_replays=0)
            try:
                return sample(*args, **kwargs)
            finally:
                self.last_stats = dict(self._stats, selected_indices=[x.cpu().tolist() for x in self._indices])
                self._indices.clear()
                self._active = False

        self._originals = [(mot, "forward", self._dense_forward), (self.model, "sample_action", sample)]
        mot.forward, self.model.sample_action = forward, sample_action
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for owner, name, original in reversed(self._originals):
            setattr(owner, name, original)
        self._originals.clear()
        self.model._visual_ffn_context_cache = None
        self._active = False
        self._indices.clear()

    def graph_stats(self):
        if self.replay is None:
            return {}
        return dict(captured=self.replay.graph is not None, capture_seconds=self.replay.capture_seconds,
                    replays=self.replay.replays)

    def close_graphs(self):
        if self._active:
            raise RuntimeError("cannot release graphs during sampling")
        self.replay = None
