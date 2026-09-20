"""Strict hybrid configuration, independent of model and evaluation libraries."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .schedule import Schedule, integer, mapping, stable_hash
from .native_config import NativeRoutingConfig


def number(value, name, low, high=None, *, open_low=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if (value <= low if open_low else value < low) or (high is not None and value > high):
        raise ValueError(f"{name} is outside its supported range")
    return float(value)


@dataclass(frozen=True)
class HybridConfig:
    schedule: Schedule
    recompute_ratio: float = 0.1
    read_mode: str = "full"
    read_ratio: float = 1.0
    selection: str = "drift"
    guidance_weight: float = 1.0
    frame_quota: str = "none"
    backend: str = "eager"
    graph_warmup: int = 3
    max_graphs: int = 8
    diagnostics: str = "counters"
    context_weight: float = 1.0
    support_seed_ratio: float = 0.1
    reuse_mode: str = "features"
    read_count: int | None = None
    native_routing: NativeRoutingConfig | None = None

    def __post_init__(self):
        if not isinstance(self.schedule, Schedule):
            raise ValueError("schedule must be a validated Schedule")
        number(self.recompute_ratio, "recompute.keep_ratio", 0, 1, open_low=True)
        number(self.read_ratio, "read.keep_ratio", 0, 1, open_low=True)
        number(self.guidance_weight, "selection.guidance_weight", 0)
        number(self.context_weight, "selection.context_weight", 0)
        number(self.support_seed_ratio, "selection.support_seed_ratio", 0, 1, open_low=True)
        enums = ((self.read_mode, ("full", "compact"), "read.mode"),
                 (self.selection, ("uniform", "drift", "action_drift", "action", "action_context", "visual_context", "native"), "selection.method"),
                 (self.frame_quota, ("balanced", "none"), "selection.frame_quota"),
                 (self.backend, ("eager", "buffered", "cuda_graph"), "execution.backend"),
                 (self.reuse_mode, ("features", "structure"), "reuse.mode"),
                 (self.diagnostics, ("counters", "trace"), "diagnostics.level"))
        for value, allowed, name in enums:
            if not isinstance(value, str) or value not in allowed:
                raise ValueError(f"{name} must be one of {allowed}")
        if self.read_mode == "full" and self.read_ratio != 1:
            raise ValueError("full read requires keep_ratio=1")
        if self.read_count is not None:
            integer(self.read_count, "read.keep_count")
            if self.read_mode != "compact" or self.read_ratio != 1:
                raise ValueError("explicit read count requires compact mode and no ratio")
        if self.read_mode == "compact":
            if self.read_count is None and self.recompute_ratio > self.read_ratio:
                raise ValueError("compact recompute budget must not exceed read budget")
            if self.frame_quota != "balanced":
                raise ValueError("compact read requires balanced frame quotas")
        if self.reuse_mode == "structure":
            if self.read_mode != "compact" or (self.read_count is None and self.recompute_ratio != self.read_ratio):
                raise ValueError("structure reuse requires compact read and equal recompute/read ratios")
            if self.selection not in ("uniform", "action", "action_context", "visual_context", "native"):
                raise ValueError("structure reuse requires a current-input selector")
        if (self.selection == "native") != (self.native_routing is not None):
            raise ValueError("native selection requires an explicit native_routing configuration")
        if self.native_routing is not None:
            if not isinstance(self.native_routing, NativeRoutingConfig):
                raise ValueError("native_routing must be a validated NativeRoutingConfig")
            if self.native_routing.separate_packing:
                if self.read_mode != "compact" or self.reuse_mode != "features" or "sparse" in self.schedule.operations:
                    raise ValueError("layerwise/pool reads require compact Dense/reuse feature schedules")
        integer(self.graph_warmup, "graph_warmup")
        integer(self.max_graphs, "max_graphs")

    @classmethod
    def from_mapping(cls, payload):
        p = mapping(payload, ("schema_version", "schedule", "recompute", "read", "selection",
                             "execution", "diagnostics", "reuse", "native_routing"), "hybrid_visual", ("schedule",))
        version = p.get("schema_version", 1)
        if type(version) is not int or version != 1:
            raise ValueError("unsupported hybrid_visual.schema_version")
        recompute = mapping(p.get("recompute", {}), ("keep_ratio",), "recompute")
        read = mapping(p.get("read", {}), ("mode", "keep_ratio", "keep_count"), "read")
        if "keep_ratio" in read and "keep_count" in read:
            raise ValueError("choose read.keep_count or read.keep_ratio")
        select = mapping(p.get("selection", {}), ("method", "guidance_weight", "frame_quota",
                                                  "context_weight", "support_seed_ratio"), "selection")
        execution = mapping(p.get("execution", {}), ("backend", "graph_warmup", "max_graphs"), "execution")
        diagnostics = mapping(p.get("diagnostics", {}), ("level",), "diagnostics")
        reuse = mapping(p.get("reuse", {}), ("mode",), "reuse")
        return cls(Schedule.from_mapping(p["schedule"]),
                   recompute.get("keep_ratio", 0.1), read.get("mode", "full"), read.get("keep_ratio", 1.0),
                   select.get("method", "drift"), select.get("guidance_weight", 1.0),
                   select.get("frame_quota", "balanced" if read.get("mode", "full") == "compact" else "none"),
                   execution.get("backend", "eager"),
                   execution.get("graph_warmup", 3), execution.get("max_graphs", 8),
                   diagnostics.get("level", "counters"), select.get("context_weight", 1.0),
                   select.get("support_seed_ratio", 0.1), reuse.get("mode", "features"), read.get("keep_count"),
                   NativeRoutingConfig.from_mapping(p["native_routing"]) if "native_routing" in p else None)

    def describe(self, *, canonical=False):
        result = dict(schema_version=1, schedule=self.schedule.describe(canonical=canonical),
                    recompute=dict(keep_ratio=float(self.recompute_ratio)),
                    read=dict(mode=self.read_mode, keep_ratio=float(self.read_ratio)),
                    selection=dict(method=self.selection, guidance_weight=float(self.guidance_weight),
                                   frame_quota=self.frame_quota),
                    execution=dict(backend=self.backend, graph_warmup=self.graph_warmup,
                                   max_graphs=self.max_graphs),
                    diagnostics=dict(level=self.diagnostics))
        # Preserve historical fingerprints when the new factor is unused.
        if self.selection in ("action", "action_context", "visual_context") or self.context_weight != 1 or self.support_seed_ratio != 0.1:
            result["selection"].update(context_weight=float(self.context_weight),
                                       support_seed_ratio=float(self.support_seed_ratio))
        if self.reuse_mode != "features":
            result["reuse"] = dict(mode=self.reuse_mode)
        if self.read_count is not None:
            result["read"] = dict(mode=self.read_mode, keep_count=self.read_count)
        if self.native_routing is not None:
            result["native_routing"] = self.native_routing.describe()
        return result

    @property
    def policy_hash(self):
        return stable_hash(self.describe(canonical=True))

    def budgets(self, length, frame_size):
        integer(length, "video length")
        integer(frame_size, "tokens_per_frame")
        if length % frame_size:
            raise ValueError("hybrid requires complete frame grids")
        q = math.ceil(length * self.recompute_ratio)
        kv = self.read_count if self.read_count is not None else math.ceil(length * self.read_ratio)
        if not 1 <= kv <= length or (self.read_mode == "compact" and q > kv):
            raise ValueError("recompute/read counts must fit the actual grid and each other")
        if self.reuse_mode == "structure" and q != kv:
            raise ValueError("structure reuse requires equal actual recompute/read counts")
        if self.read_mode == "compact" and kv < length // frame_size:
            raise ValueError("read budget cannot cover every frame")
        # Only read routes require frame coverage. Query selection may use any rows,
        # including no observed-frame rows, preserving the old drift-refresh behavior.
        return q, kv
