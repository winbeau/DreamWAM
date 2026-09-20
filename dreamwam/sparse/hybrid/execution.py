"""Transformer math only: no schedule, selection, or request-cache mutation."""

import torch

from dreamwam.layers import scaled_dot_product_attention


def tensor_state(state, *, video=False):
    keys = ("tokens", "freqs", "time_modulation", "context", "context_mask")
    if video:
        keys += ("tokens_per_frame",)
    return {key: state[key] for key in keys}


def selected_video_state(state, index):
    result = tensor_state(state, video=True)
    result["tokens"] = state["tokens"].index_select(1, index)
    freqs = state["freqs"]
    result["freqs"] = (tuple(part.index_select(0, index) for part in freqs)
                       if isinstance(freqs, tuple) else freqs.index_select(0, index))
    if state["time_modulation"].ndim == 4:
        result["time_modulation"] = state["time_modulation"].index_select(1, index)
    if state["context_mask"].ndim == 3:
        result["context_mask"] = state["context_mask"].index_select(1, index)
    return result


class HybridExecutor:
    def __init__(self, model, native_forward):
        self.model = model
        self.native_forward = native_forward

    def router_output(self, **result):
        router = self.model.world_residual
        return dict(**result, gates={k: list(v) for k, v in router._runtime_gates.items()},
                    previews={k: list(v) for k, v in router._runtime_local_preview_tokens.items()})

    def dense(self, *, video_state, action_state):
        mot = self.model.mot
        router = self.model.world_residual
        router.reset_runtime()
        original = mot._attention_input
        layers = {id(block): i for i, block in enumerate(mot.video_expert.blocks)}
        captured = {}

        def capture(block, *args, **kwargs):
            result = original(block, *args, **kwargs)
            if id(block) in layers:
                captured[layers[id(block)]] = dict(k=result[1], v=result[2])
            return result

        mot._attention_input = capture
        try:
            output = self.native_forward(video_state=video_state, action_state=action_state,
                                         residual_injection=router, sparse=None)
        finally:
            mot._attention_input = original
        if len(captured) != mot.num_layers:
            raise RuntimeError("dense anchor did not capture every visual layer")
        return self.router_output(**output, kv=[captured[i] for i in range(mot.num_layers)])

    def sparse(self, *, video_state, action_state, kv, query_slots, mask):
        mot = self.model.mot
        router = self.model.world_residual
        router.reset_runtime()
        video, action = video_state["tokens"], action_state["tokens"]
        keep = video.shape[1]
        updated, packed = [], []
        for layer in range(mot.num_layers):
            vb, ab = mot.video_expert.blocks[layer], mot.action_expert.blocks[layer]
            vio = mot._attention_input(vb, video, video_state["freqs"], video_state["time_modulation"])
            aio = mot._attention_input(ab, action, action_state["freqs"], action_state["time_modulation"])
            key = vio[1] if kv is None else kv[layer]["k"].index_copy(1, query_slots, vio[1])
            value = vio[2] if kv is None else kv[layer]["v"].index_copy(1, query_slots, vio[2])
            mixed = scaled_dot_product_attention(
                torch.cat((vio[0], aio[0]), dim=1), torch.cat((key, aio[1]), dim=1),
                torch.cat((value, aio[2]), dim=1), mot.num_heads, mask)
            video = mot._post_attention(vb, vio[3], mixed[:, :keep], *vio[4:],
                                        video_state["context"], video_state["context_mask"])
            action = mot._post_attention(ab, aio[3], mixed[:, keep:], *aio[4:],
                                         action_state["context"], action_state["context_mask"])
            video = router(layer, video, video_state["context"], video_state["context_mask"])
            packed.append(dict(k=key, v=value))
            updated.append(dict(k=vio[1], v=vio[2]))
        return self.router_output(video=video, action=action, kv=packed, updates=updated)

    def fresh(self, *, video_state, action_state, mask):
        """Same transformer math, but every read key is computed THIS step.

        No cached K/V is passed in, including through CUDA graph buffers.
        """
        return self.sparse(video_state=video_state, action_state=action_state,
                           kv=None, query_slots=None, mask=mask)

    def reuse(self, *, action_state, kv, mask):
        mot = self.model.mot
        action = action_state["tokens"]
        for layer in range(mot.num_layers):
            block = mot.action_expert.blocks[layer]
            io = mot._attention_input(block, action, action_state["freqs"], action_state["time_modulation"])
            mixed = scaled_dot_product_attention(
                io[0], torch.cat((kv[layer]["k"], io[1]), dim=1),
                torch.cat((kv[layer]["v"], io[2]), dim=1), mot.num_heads, mask)
            action = mot._post_attention(block, io[3], mixed, *io[4:],
                                         action_state["context"], action_state["context_mask"])
        return dict(action=action)
