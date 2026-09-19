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

from .attention import sparse_joint_attention
from .config import SparseConfig
from .layout import TokenLayout, build_token_layout, video_legality
from .routing import Route, build_route

__all__ = [
    "Route",
    "SparseConfig",
    "TokenLayout",
    "build_route",
    "build_token_layout",
    "sparse_joint_attention",
    "video_legality",
]
