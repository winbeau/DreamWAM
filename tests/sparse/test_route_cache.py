"""Integration tests for the route-cache wiring in :meth:`JointMoT._joint_self_attention`.

The pure scope functions in ``dreamwam.sparse.runtime`` are unit tested, but the wiring that
consumes them was not: whether the cache is actually populated once per scope, actually read
back on later layers, actually cleared between sampling calls, and whether the dense prefix
really runs dense. A defect there silently changes the executed route, so it is verified here
without a checkpoint or a GPU by binding the method to a stand-in object that provides only
the attributes it touches.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from dreamwam.mot import JointMoT
from dreamwam.sparse import SparseConfig

TOKENS_PER_FRAME = 4
NUM_FRAMES = 3
VIDEO_TOKENS = TOKENS_PER_FRAME * NUM_FRAMES
ACTION_TOKENS = 3
NUM_HEADS = 2
HEAD_DIM = 4
WIDTH = NUM_HEADS * HEAD_DIM
BLOCK_SIZE = 2


def make_mot(config: SparseConfig) -> JointMoT:
    """A JointMoT bound to just enough state for _joint_self_attention to run."""
    mot = object.__new__(JointMoT)
    mot.num_heads = NUM_HEADS
    mot._layout_cache = {}
    mot._route_cache = {}
    mot.reset_sparse_diagnostics()
    # Only `.blocks[0].self_attn.q.weight.device` is touched, to place the layout.
    mot.video_expert = SimpleNamespace(
        blocks=[SimpleNamespace(self_attn=SimpleNamespace(q=SimpleNamespace(weight=torch.zeros(1))))]
    )
    return mot


def inputs(seed: int = 0):
    generator = torch.Generator().manual_seed(seed)

    def rand(length):
        return torch.randn(1, length, WIDTH, generator=generator)

    video = (rand(VIDEO_TOKENS), rand(VIDEO_TOKENS), rand(VIDEO_TOKENS))
    action = (rand(ACTION_TOKENS), rand(ACTION_TOKENS), rand(ACTION_TOKENS))
    mask = torch.zeros(VIDEO_TOKENS + ACTION_TOKENS, VIDEO_TOKENS + ACTION_TOKENS, dtype=torch.bool)
    mask[:TOKENS_PER_FRAME, :TOKENS_PER_FRAME] = True
    mask[TOKENS_PER_FRAME:, :] = True
    mask[VIDEO_TOKENS:, VIDEO_TOKENS:] = True
    return video, action, mask


def call(mot: JointMoT, config: SparseConfig, *, step: int, layer: int, data=None):
    video, action, mask = data if data is not None else inputs()
    return mot._joint_self_attention(
        video_io=video,
        action_io=action,
        attention_mask=mask,
        video_length=VIDEO_TOKENS,
        tokens_per_frame=TOKENS_PER_FRAME,
        sparse=config,
        step_index=step,
        num_steps=10,
        layer_index=layer,
    )


def step_config(**overrides) -> SparseConfig:
    payload = {
        "enabled": True,
        "selection": "av",
        "backend": "masked",
        "block_size": BLOCK_SIZE,
        "future_ratio": 0.5,
        "anchor_refresh": "step",
    }
    payload.update(overrides)
    return SparseConfig.from_mapping(payload)


def test_step_scope_builds_one_route_per_step_and_reuses_it():
    config = step_config()
    mot = make_mot(config)
    data = inputs()
    for layer in range(4):
        call(mot, config, step=0, layer=layer, data=data)
    assert mot.sparse_diagnostics["calls"] == 4
    assert mot.sparse_diagnostics["anchor_builds"] == 1, "route must be built once per step"
    assert len(mot._route_cache) == 1
    # The next step builds its own route; the previous entry stays until the call ends.
    for layer in range(4):
        call(mot, config, step=1, layer=layer, data=data)
    assert mot.sparse_diagnostics["anchor_builds"] == 2
    assert len(mot._route_cache) == 2


def test_request_scope_builds_once_for_the_whole_sampling_call():
    """The regression this test exists for: request scope must not rebuild per step."""
    config = step_config(anchor_refresh="request")
    mot = make_mot(config)
    data = inputs()
    for step in range(10):
        for layer in range(3):
            call(mot, config, step=step, layer=layer, data=data)
    assert mot.sparse_diagnostics["calls"] == 30
    assert mot.sparse_diagnostics["anchor_builds"] == 1, (
        "anchor_refresh='request' rebuilt during the call, so it degraded to 'step' scope"
    )
    assert len(mot._route_cache) == 1


def test_layer_scope_rebuilds_every_layer():
    config = step_config(anchor_refresh="layer")
    mot = make_mot(config)
    data = inputs()
    for layer in range(4):
        call(mot, config, step=0, layer=layer, data=data)
    assert mot.sparse_diagnostics["anchor_builds"] == 4
    assert mot._route_cache == {}


def test_reused_route_is_identical_across_layers_of_one_step():
    """Layers in one step must execute the same key set, otherwise reuse is not reuse."""
    config = step_config()
    mot = make_mot(config)
    data = inputs()
    call(mot, config, step=0, layer=0, data=data)
    first = mot._route_cache[(0,)]
    call(mot, config, step=0, layer=1, data=data)
    second = mot._route_cache[(0,)]
    assert first is second
    assert torch.equal(first.keys, second.keys)


def test_dense_prefix_layers_run_without_building_a_route():
    config = step_config(anchor_layer=2)
    mot = make_mot(config)
    data = inputs()
    for layer in range(2):
        call(mot, config, step=0, layer=layer, data=data)
    assert mot.sparse_diagnostics["anchor_builds"] == 0
    assert mot._route_cache == {}, "the dense prefix must not populate the route cache"
    call(mot, config, step=0, layer=2, data=data)
    assert mot.sparse_diagnostics["anchor_builds"] == 1


def test_dense_prefix_output_equals_the_disabled_path():
    """A dense-prefix layer must be bit-identical to running with sparse switched off."""
    config = step_config(anchor_layer=1)
    mot = make_mot(config)
    video, action, mask = inputs()
    sparse_out = mot._joint_self_attention(
        video_io=video,
        action_io=action,
        attention_mask=mask,
        video_length=VIDEO_TOKENS,
        tokens_per_frame=TOKENS_PER_FRAME,
        sparse=config,
        step_index=0,
        num_steps=10,
        layer_index=0,
    )
    dense_out = mot._joint_self_attention(
        video_io=video,
        action_io=action,
        attention_mask=mask,
        video_length=VIDEO_TOKENS,
        tokens_per_frame=TOKENS_PER_FRAME,
        sparse=SparseConfig(),
        step_index=0,
        num_steps=10,
        layer_index=0,
    )
    assert torch.equal(sparse_out, dense_out)


def test_reset_clears_the_route_cache_between_sampling_calls():
    config = step_config(anchor_refresh="request")
    mot = make_mot(config)
    data = inputs()
    for layer in range(3):
        call(mot, config, step=0, layer=layer, data=data)
    assert len(mot._route_cache) == 1
    mot.reset_sparse_diagnostics()
    assert mot._route_cache == {}, "a stale route must not survive into the next request"
    assert mot.sparse_diagnostics["anchor_builds"] == 0


def test_disabled_sparse_never_touches_the_cache():
    config = SparseConfig()
    mot = make_mot(config)
    data = inputs()
    for layer in range(3):
        call(mot, config, step=0, layer=layer, data=data)
    assert mot._route_cache == {}
    assert mot.sparse_diagnostics["calls"] == 0
