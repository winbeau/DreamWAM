"""Sparse-WAM inference components for DreamWAM Joint.

The package is inference-only and off by default.  Three pieces, matching the paper's
module split:

* :mod:`dreamwam.sparse.layout`   - how a video/action sequence is partitioned into frames
  and key blocks, and which video key a video query may legally see.
* :mod:`dreamwam.sparse.av`       - the action-query attention mass that locates action
  anchors (A->V).
* :mod:`dreamwam.sparse.routing`  - turns anchors plus visual statistics into the executed
  VV route (M2).
* :mod:`dreamwam.sparse.attention` - executes the route (M3).

Nothing here changes A->V or A->A: action queries keep the joint softmax over
``[video; action]`` keys, and their output is mathematically identical to the shipped
fused call.
"""

from importlib import import_module

__all__ = [
    "Route",
    "SparseConfig",
    "TokenLayout",
    "build_route",
    "build_token_layout",
    "sparse_joint_attention",
    "video_legality",
]

_EXPORTS = {
    "sparse_joint_attention": "attention", "SparseConfig": "config",
    "TokenLayout": "layout", "build_token_layout": "layout", "video_legality": "layout",
    "Route": "routing", "build_route": "routing",
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module("." + _EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
