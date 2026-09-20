"""One-request visual state; executors return values and never commit this state."""

import torch


class VisualState:
    def __init__(self):
        self.clear()

    def clear(self):
        self.kv = []
        self.packed = []
        self.output = None
        self.reference = None
        self.route = None
        self.updated_at = None
        self.route_step = None
        self.layout = None
        self.full_mask = None
        self.action_mask = None
        self.action_indices = None

    def check_layout(self, video, frame_size, action_length):
        layout = (tuple(video.shape), video.dtype, video.device, frame_size, action_length)
        if self.layout is not None and layout != self.layout:
            raise ValueError("visual/action layout changed within one request")
        self.layout = layout

    def commit_dense(self, result, current, step):
        self.kv = result["kv"]
        self.output = result["video"]
        self.reference = current.clone()
        self.updated_at = torch.full((current.shape[1],), step, dtype=torch.long, device=current.device)

    def commit_sparse(self, result, current, query, route, step, *, full_read):
        if full_read:
            self.kv = result["kv"]
        else:
            # The entire transformer call succeeded before touching canonical state.
            for cache, fresh in zip(self.kv, result["updates"]):
                for name in ("k", "v"):
                    cache[name].index_copy_(1, query, fresh[name])
        self.output = self.output.index_copy(1, query, result["video"])
        self.reference.index_copy_(1, query, current.index_select(1, query))
        self.updated_at.index_fill_(0, query, step)
        self.packed = result["kv"]
        self.route = route
        self.route_step = step

    def pack(self, route, step, *, full_read):
        self.route = route
        self.route_step = step
        self.packed = (self.kv if full_read else
                       [{name: value.index_select(1, route) for name, value in layer.items()}
                        for layer in self.kv])
