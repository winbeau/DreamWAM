"""Bounded memoization of frozen text encoding; shared Dense/Sparse control.

Only exact prompt batches are keys. Camera/state inputs are never cached.
Outputs are copied on every call, and encoder/tokenizer changes invalidate
the entries. This common optimization must be enabled on both timing arms.
"""

from collections import OrderedDict
from collections.abc import Mapping

import torch


def prompt_cache_options(payload):
    """Opt-in configuration; an omitted option preserves uncached inference."""
    if payload is None:
        return None
    if not isinstance(payload, Mapping) or set(payload) - {"capacity"}:
        raise ValueError("prompt_cache accepts only capacity")
    capacity = payload.get("capacity", 8)
    if type(capacity) is not int or capacity < 1:
        raise ValueError("prompt_cache.capacity must be a positive integer")
    return {"capacity": capacity}


class PromptEncodingCache:
    def __init__(self, encoder, *, capacity=8):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        self.encoder, self.capacity = encoder, capacity
        self.entries = OrderedDict()
        self.signature = None
        self.hits = self.misses = self.invalidations = 0
        self.last_hit = False

    def _signature(self):
        model, tokenizer = self.encoder.model, self.encoder.tokenizer
        if model.training or any(p.requires_grad for p in model.parameters()):
            raise ValueError("prompt caching requires a frozen eval text encoder")
        return (id(model), id(tokenizer), id(getattr(tokenizer, "tokenizer", None)),
                getattr(tokenizer, "seq_len", None), getattr(tokenizer, "clean", None),
                self.encoder.device, self.encoder.dtype,
                tuple((name, id(p), p._version, p.device, p.dtype)
                      for name, p in model.named_parameters()),
                tuple((name, id(p), p._version, p.device, p.dtype)
                      for name, p in model.named_buffers()))

    @torch.no_grad()
    def __call__(self, prompts):
        if not isinstance(prompts, list) or not prompts or not all(isinstance(p, str) and p for p in prompts):
            raise ValueError("prompts must be a non-empty list of strings")
        signature = self._signature()
        if signature != self.signature:
            if self.signature is not None:
                self.invalidations += 1
            self.entries.clear()
            self.signature = signature
        key = tuple(prompts)
        self.last_hit = key in self.entries
        if self.last_hit:
            self.hits += 1
            result = self.entries.pop(key)
        else:
            self.misses += 1
            result = self.encoder(prompts)
            if not (isinstance(result, tuple) and len(result) == 2 and
                    all(isinstance(t, torch.Tensor) for t in result)):
                raise TypeError("expected context/mask tensor pair")
            result = tuple(t.detach().clone() for t in result)
        self.entries[key] = result
        while len(self.entries) > self.capacity:
            self.entries.popitem(last=False)
        return tuple(t.clone() for t in result)

    def clear(self):
        self.entries.clear()
        self.signature = None
        self.last_hit = False

    def stats(self):
        return dict(hits=self.hits, misses=self.misses, invalidations=self.invalidations,
                    entries=len(self.entries), last_hit=self.last_hit, capacity=self.capacity)
