from types import SimpleNamespace

import pytest
import torch

from dreamwam.sparse.prompt_cache import PromptEncodingCache


class Encoder:
    def __init__(self):
        self.model = torch.nn.Linear(1, 2).eval().requires_grad_(False)
        self.tokenizer = SimpleNamespace(seq_len=128, clean="whitespace")
        self.device, self.dtype = torch.device("cpu"), torch.float32
        self.calls = 0

    def __call__(self, prompts):
        self.calls += 1
        values = torch.tensor([[sum(map(ord, p))] for p in prompts], dtype=self.dtype)
        return self.model(values), torch.ones(len(prompts), 2, dtype=torch.bool)


@torch.no_grad()
def test_exact_batches_eviction_and_output_ownership():
    encoder = Encoder()
    cache = PromptEncodingCache(encoder, capacity=2)
    first = cache(["red cup"])
    original = first[0].clone()
    first[0].zero_(); first[1].zero_()
    repeated = cache(["red cup"])
    assert torch.equal(repeated[0], original) and repeated[1].all()
    assert encoder.calls == 1 and cache.last_hit
    cache(["blue cup"]); cache(["red cup"]); cache(["green cup"])
    cache(["blue cup"])
    assert encoder.calls == 4 and len(cache.entries) == 2
    cache(["red cup", "blue cup"]); cache(["blue cup", "red cup"])
    assert not cache.last_hit


@torch.no_grad()
def test_weight_and_tokenizer_changes_invalidate_and_rng_is_unchanged():
    encoder = Encoder(); cache = PromptEncodingCache(encoder)
    rng = torch.get_rng_state().clone()
    before = cache(["cup"])[0]
    with torch.no_grad():
        encoder.model.weight.add_(1)
    after = cache(["cup"])[0]
    assert not torch.equal(before, after) and cache.invalidations == 1
    encoder.tokenizer.seq_len = 64
    cache(["cup"])
    assert cache.invalidations == 2 and not cache.last_hit
    assert torch.equal(rng, torch.get_rng_state())
    cache.clear(); assert not cache.entries
    encoder.model.train()
    with pytest.raises(ValueError, match="frozen eval"):
        cache(["cup"])


@pytest.mark.parametrize("prompts", [[], [""], [1], "cup", ("cup",)])
def test_invalid_prompts_cannot_hit_cache(prompts):
    with pytest.raises(ValueError):
        PromptEncodingCache(Encoder())(prompts)
