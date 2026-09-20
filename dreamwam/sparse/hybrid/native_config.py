"""Explicit native-anchor read signals and independent recompute selection."""

from dataclasses import asdict, dataclass
import math

from .schedule import integer, mapping


SIGNALS = ("uniform", "action", "value_norm", "value_action", "dynamic", "visual_context", "action_context", "fusion")


@dataclass(frozen=True)
class NativeRoutingConfig:
    signal: str = "value_action"
    layers: tuple[int, ...] | None = None
    heads: int = 4
    layerwise: bool = False
    recompute: str = "drift"
    observed: str = "score"
    packing: str = "hard"
    refine_fraction: float = 0.25
    multiplicity: str = "count"
    weights: tuple[tuple[str, float], ...] = ()

    def __post_init__(self):
        for value, choices, name in (
            (self.signal, SIGNALS, "signal"), (self.recompute, ("uniform", "drift", "anchor"), "recompute"),
            (self.observed, ("score", "uniform", "full"), "observed"),
            (self.packing, ("hard", "pool"), "packing"), (self.multiplicity, ("count", "unit"), "multiplicity")):
            if value not in choices:
                raise ValueError(f"native_routing.{name} must be one of {choices}")
        integer(self.heads, "native_routing.heads")
        if type(self.layerwise) is not bool:
            raise ValueError("native_routing.layerwise must be boolean")
        if self.layers is not None:
            if not isinstance(self.layers, tuple) or not self.layers or len(set(self.layers)) != len(self.layers):
                raise ValueError("native_routing.layers must be a nonempty unique tuple")
            for layer in self.layers:
                integer(layer, "native_routing.layer", 0)
        if self.layerwise and self.layers is not None:
            raise ValueError("layerwise routing scores every layer; do not specify a partial depth sample")
        if (type(self.refine_fraction) not in (int, float) or not math.isfinite(self.refine_fraction) or
            not 0 <= self.refine_fraction <= 1):
            raise ValueError("native_routing.refine_fraction must lie in [0,1]")
        if self.packing == "pool" and self.observed != "full":
            raise ValueError("background pooling retains the complete observed frame explicitly")
        if not isinstance(self.weights, tuple) or len(dict(self.weights)) != len(self.weights):
            raise ValueError("fusion weights must be an immutable unique set")
        for name, value in self.weights:
            if (name not in SIGNALS[1:-1] or type(value) not in (int, float) or
                not math.isfinite(value) or value < 0):
                raise ValueError("fusion weights require named finite nonnegative proxy weights")
        if self.signal == "fusion":
            if not self.weights or not any(value > 0 for _, value in self.weights):
                raise ValueError("fusion requires explicit externally calibrated weights")
        elif self.weights:
            raise ValueError("weights only apply to the explicit fusion ablation")

    @classmethod
    def from_mapping(cls, value):
        payload = mapping(value, tuple(cls.__dataclass_fields__), "native_routing")
        if payload.get("layers") is not None:
            payload["layers"] = tuple(payload["layers"])
        if "weights" in payload:
            weights = mapping(payload["weights"], SIGNALS[1:-1], "native_routing.weights")
            payload["weights"] = tuple(sorted(weights.items()))
        return cls(**payload)

    def describe(self):
        result = asdict(self)
        result["layers"] = None if self.layers is None else list(self.layers)
        result["weights"] = dict(self.weights)
        return result

    def sampled_layers(self, depth):
        if self.layerwise:
            return tuple(range(depth))
        result = self.layers or tuple(sorted({i * (depth - 1) // 3 for i in range(4)}))
        if any(layer >= depth for layer in result):
            raise ValueError("native score layer exceeds model depth")
        return result

    @property
    def separate_packing(self):
        return self.layerwise or self.packing == "pool"
