"""Per-request runtime state for Sparse-WAM: route scope keys and refresh policy.

Kept as pure functions so the amortization decisions can be tested without a model, a
checkpoint or a GPU.  Two of these decisions were wrong in the first implementation:
``anchor_refresh="request"`` rebuilt the route at layer 0 of every denoising step and so
silently degraded to ``"step"``, and ``anchor_layer > 0`` was ignored on a cache hit.
"""

from __future__ import annotations

from .config import SparseConfig


def route_scope_key(config: SparseConfig, step_index: int) -> tuple[int, ...] | None:
    """Cache key identifying the lifetime of one route, or ``None`` to rebuild per layer.

    ``layer``   -> ``None`` (no caching; the original per-layer behaviour)
    ``step``    -> one route per denoising step, shared by every layer in that step
    ``request`` -> one route for the whole sampling call, shared by every layer and step
    """
    if config.anchor_refresh == "layer":
        return None
    if config.anchor_refresh == "step":
        return (int(step_index),)
    return (0,)


def is_dense_prefix(config: SparseConfig, layer_index: int) -> bool:
    """Whether this layer runs the original dense path while the route is not yet built.

    Only meaningful when routes are amortized: layers before ``anchor_layer`` have no route
    to reuse, and running them dense keeps the anchor signal taken from the layer the caller
    asked for instead of from whatever layer happens to be first.
    """
    if config.anchor_refresh == "layer":
        return False
    return layer_index < config.anchor_layer


def should_build_route(
    config: SparseConfig,
    layer_index: int,
    cached: object | None,
) -> bool:
    """Whether the route must be built here rather than reused."""
    if config.anchor_refresh == "layer":
        return True
    if is_dense_prefix(config, layer_index):
        return False
    return cached is None
