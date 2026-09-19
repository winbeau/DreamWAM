"""Strict configuration schema for Sparse-WAM inference.

Unknown keys are rejected rather than ignored: a typo in an experiment YAML must not
silently produce a dense run that is then reported as a sparse one.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

#: How the future-key part of the route is chosen (M2).
#: ``all``       keep every future block (full-budget control)
#: ``av``        top blocks by action-query attention mass (A->V anchors only)
#: ``av_context`` anchors re-ranked with a cheap VV affinity term
#: ``recency``   most recent blocks only, no action information (visual-only baseline)
#: ``uniform``   evenly spaced blocks, no action information (visual-only baseline)
SELECTIONS = ("all", "av", "av_context", "recency", "uniform")

#: ``masked`` keeps the shipped full-length attention call and only restricts the mask,
#: which is numerically exact but buys no speed; ``gather`` selects keys into a shorter
#: tensor.  The measured cost of both is recorded in docs/implementation/.
BACKENDS = ("masked", "gather")

FALLBACKS = ("recency", "uniform", "all")

#: Route/anchor refresh scope. ``layer`` keeps the original per-layer behaviour; the other
#: two amortize the anchor extraction over a denoising step or a whole request.
ANCHOR_REFRESH = ("layer", "step", "request")


@dataclass(frozen=True)
class SparseConfig:
    """Validated, frozen sparse-attention configuration for one evaluation."""

    enabled: bool = False
    selection: str = "av_context"
    backend: str = "masked"
    block_size: int = 14
    future_ratio: float = 1.0
    #: Fraction of conditioning-frame keys kept for *future-frame* queries.  Keeping the whole
    #: conditioning frame was the original structural floor because it is what the action
    #: currently observes; it is a knob rather than a floor because the video branch tolerates
    #: far more compression than the action branch does.  Frame-0 queries always retain dense
    #: access to their own frame, which is cheap (98x98) and required by the native mask.
    conditional_keep_ratio: float = 1.0
    context_weight: float = 1.0
    min_anchor_mass: float = 0.0
    fallback: str = "recency"
    num_stages: int = 1
    #: How often the anchor signal and the route are recomputed.  ``layer`` recomputes at
    #: every layer, ``step`` computes once per denoising step and reuses, ``request``
    #: computes once for the whole sampling call.  Measured cost matters: the action-anchor
    #: extraction is more expensive per layer-step than the attention it guides, so
    #: amortizing it over layers is what turns a loss into a gain (see docs/analysis).
    anchor_refresh: str = "layer"
    #: Layer that builds the route when ``anchor_refresh`` is not ``layer``.
    anchor_layer: int = 0
    #: Layers that execute the routed path; ``None`` means every layer.  M1 calibration
    #: needs single-layer interventions ("what happens if only this layer is restricted"),
    #: which is also the only way to attribute an action change to a specific layer.
    sparse_layers: tuple[int, ...] | None = None
    #: Per-head kept future blocks; length must equal the model's head count.  ``None``
    #: derives the count from ``future_ratio``.  Produced by M1 calibration.
    head_blocks: tuple[int, ...] | None = None
    #: ``[num_stages][num_heads]`` kept future blocks, overriding ``head_blocks``.
    stage_blocks: tuple[tuple[int, ...], ...] | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "SparseConfig":
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise TypeError(
                f"sparse options must be a mapping, got {type(payload).__name__}"
            )
        fields = {field for field in cls.__dataclass_fields__}
        unknown = sorted(set(payload) - fields)
        if unknown:
            raise ValueError(
                f"unknown sparse option(s): {unknown}; known options are {sorted(fields)}"
            )
        normalized: dict[str, Any] = {}
        for key, value in payload.items():
            if key in {"head_blocks", "stage_blocks", "sparse_layers"}:
                normalized[key] = _nested_int_tuple(value, key)
            else:
                normalized[key] = value
        config = replace(cls(), **normalized)
        config.validate()
        return config

    def validate(self) -> None:
        if self.selection not in SELECTIONS:
            raise ValueError(
                f"selection must be one of {SELECTIONS}, got {self.selection!r}"
            )
        if self.backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {self.backend!r}")
        if self.anchor_refresh not in ANCHOR_REFRESH:
            raise ValueError(
                f"anchor_refresh must be one of {ANCHOR_REFRESH}, got "
                f"{self.anchor_refresh!r}"
            )
        if self.anchor_layer < 0:
            raise ValueError(
                f"anchor_layer must be non-negative, got {self.anchor_layer}"
            )
        if self.sparse_layers is not None:
            if not self.sparse_layers:
                raise ValueError(
                    "sparse_layers must be non-empty when set; omit it to route every layer"
                )
            if any(index < 0 for index in self.sparse_layers):
                raise ValueError("sparse_layers entries must be non-negative")
            if len(set(self.sparse_layers)) != len(self.sparse_layers):
                raise ValueError("sparse_layers must not repeat a layer")
        if self.fallback not in FALLBACKS:
            raise ValueError(
                f"fallback must be one of {FALLBACKS}, got {self.fallback!r}"
            )
        if self.block_size < 1:
            raise ValueError(f"block_size must be positive, got {self.block_size}")
        if not 0.0 <= self.future_ratio <= 1.0:
            raise ValueError(
                f"future_ratio must be in [0, 1], got {self.future_ratio}"
            )
        if not 0.0 <= self.conditional_keep_ratio <= 1.0:
            raise ValueError(
                "conditional_keep_ratio must be in [0, 1], got "
                f"{self.conditional_keep_ratio}"
            )
        if self.context_weight < 0.0:
            raise ValueError(
                f"context_weight must be non-negative, got {self.context_weight}"
            )
        if not 0.0 <= self.min_anchor_mass <= 1.0:
            raise ValueError(
                f"min_anchor_mass must be in [0, 1], got {self.min_anchor_mass}"
            )
        if self.num_stages < 1:
            raise ValueError(f"num_stages must be positive, got {self.num_stages}")
        if self.head_blocks is not None and any(
            value < 0 for value in self.head_blocks
        ):
            raise ValueError("head_blocks entries must be non-negative")
        if self.stage_blocks is not None:
            if len(self.stage_blocks) != self.num_stages:
                raise ValueError(
                    "stage_blocks must have num_stages rows, got "
                    f"{len(self.stage_blocks)} for num_stages={self.num_stages}"
                )
            widths = {len(row) for row in self.stage_blocks}
            if len(widths) > 1 or (widths and 0 in widths):
                raise ValueError("every stage_blocks row must be non-empty and equal")
            for row in self.stage_blocks:
                if any(value < 0 for value in row):
                    raise ValueError("stage_blocks entries must be non-negative")
        if self.head_blocks is not None and self.stage_blocks is not None:
            if len(self.head_blocks) != len(self.stage_blocks[0]):
                raise ValueError(
                    "head_blocks and stage_blocks must agree on the head count"
                )

    def stage_of(self, step_index: int, num_steps: int) -> int:
        """Map a denoising step to its stage bucket."""
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")
        if step_index < 0 or step_index >= num_steps:
            raise ValueError(
                f"step_index {step_index} outside [0, {num_steps}) for {num_steps} steps"
            )
        if self.num_stages == 1:
            return 0
        return min(self.num_stages - 1, step_index * self.num_stages // num_steps)

    def blocks_for(self, *, num_heads: int, stage: int, num_future_blocks: int) -> tuple[int, ...]:
        """Kept future-block count per head, clamped to the available blocks."""
        if self.stage_blocks is not None:
            row = self.stage_blocks[stage]
        elif self.head_blocks is not None:
            row = self.head_blocks
        else:
            keep = int(round(self.future_ratio * num_future_blocks))
            row = (keep,) * num_heads
        if len(row) != num_heads:
            raise ValueError(
                f"budget has {len(row)} entries but the model has {num_heads} heads"
            )
        return tuple(min(int(value), num_future_blocks) for value in row)

    def describe(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "selection": self.selection,
            "backend": self.backend,
            "block_size": self.block_size,
            "future_ratio": self.future_ratio,
            "conditional_keep_ratio": self.conditional_keep_ratio,
            "context_weight": self.context_weight,
            "min_anchor_mass": self.min_anchor_mass,
            "fallback": self.fallback,
            "num_stages": self.num_stages,
            "anchor_refresh": self.anchor_refresh,
            "anchor_layer": self.anchor_layer,
            "sparse_layers": (
                None if self.sparse_layers is None else list(self.sparse_layers)
            ),
            "head_blocks": None if self.head_blocks is None else list(self.head_blocks),
            "stage_blocks": (
                None
                if self.stage_blocks is None
                else [list(row) for row in self.stage_blocks]
            ),
        }


def _nested_int_tuple(value: Any, name: str) -> tuple:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if value and isinstance(value[0], (list, tuple)):
            return tuple(tuple(int(item) for item in row) for row in value)
        return tuple(int(item) for item in value)
    raise TypeError(f"{name} must be a sequence of integers")


def config_hash(config: SparseConfig) -> str:
    """Stable digest of the effective sparse configuration."""
    import hashlib
    import json

    payload = json.dumps(config.describe(), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:16]
