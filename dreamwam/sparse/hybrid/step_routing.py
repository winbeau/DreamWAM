"""Causal step decisions before visual attention/MLPs, with a hard Q-row cap.

Relative drift is an uncalibrated proxy for refresh need, not an action error
estimate. The mandatory anchor and every later refresh count against the cap.
"""

from dataclasses import dataclass
import math
import time

from .schedule import integer, mapping


@dataclass(frozen=True)
class StepRouterConfig:
    sparse_drift: float = 0.05
    dense_drift: float = 0.35
    max_reuse_steps: int = 4
    max_feature_age: int = 8
    extra_dense_budget: float = 1.0

    def __post_init__(self):
        for name in ("sparse_drift", "dense_drift", "extra_dense_budget"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(name + " must be finite and nonnegative")
        if self.dense_drift <= self.sparse_drift:
            raise ValueError("dense_drift must exceed sparse_drift")
        integer(self.max_reuse_steps, "max_reuse_steps")
        integer(self.max_feature_age, "max_feature_age")

    @classmethod
    def from_mapping(cls, payload):
        p = mapping(payload, ("kind", "sparse_drift", "dense_drift", "max_reuse_steps",
                              "max_feature_age", "extra_dense_budget"), "step_router", ("kind",))
        if p.pop("kind") != "adaptive":
            raise ValueError("step_router.kind must be adaptive")
        return cls(**p)

    def describe(self):
        return dict(kind="adaptive", sparse_drift=float(self.sparse_drift), dense_drift=float(self.dense_drift),
                    max_reuse_steps=self.max_reuse_steps, max_feature_age=self.max_feature_age,
                    extra_dense_budget=float(self.extra_dense_budget))

    def decide(self, *, step, score, route_age, feature_age, remaining, full_rows, sparse_rows):
        if step == 0:
            return "dense", "mandatory_chunk_anchor"
        if not math.isfinite(score) or score < 0:
            raise FloatingPointError("step routing drift must be finite and nonnegative")
        wants_dense = score >= self.dense_drift or feature_age >= self.max_feature_age
        wants_sparse = score >= self.sparse_drift or route_age >= self.max_reuse_steps
        if wants_dense and remaining >= full_rows:
            return "dense", "large_drift" if score >= self.dense_drift else "feature_age_limit"
        if (wants_dense or wants_sparse) and remaining >= sparse_rows:
            return "sparse", ("dense_unaffordable_sparse" if wants_dense else
                               "input_drift" if score >= self.sparse_drift else "reuse_age_limit")
        if wants_dense or wants_sparse:
            return "reuse", "query_budget_exhausted"
        return "reuse", "small_drift_and_young_cache"


def observe_and_route(config, video, state, *, step, num_steps, query_rows, spent_rows):
    """One input-drift reduction; no Q/K probes, visual layer, or future tensor."""
    started = time.perf_counter()
    batch, length = video.shape[:2]
    cap = min(num_steps * length, length + math.floor(config.extra_dense_budget * length))
    if step == 0:
        score, feature_age, route_age = 0.0, 0, 0
    else:
        import torch
        from ..reuse import token_drift
        if state.reference is None or state.route is None:
            raise RuntimeError("step routing requires this chunk's anchor")
        drift = token_drift(video.float(), state.reference.float()).mean()
        age = (step - state.updated_at.index_select(0, state.route)).max()
        score, feature_age = torch.stack((drift, age.float())).detach().cpu().tolist()
        feature_age = int(feature_age)
        route_age = step - state.route_step
    operation, reason = config.decide(step=step, score=score, route_age=route_age,
        feature_age=feature_age, remaining=cap - spent_rows, full_rows=length, sparse_rows=query_rows)
    return operation, dict(reason=reason, input_drift_mean=score, route_age_before=route_age,
        retained_feature_age_max_before=feature_age, query_row_cap=cap,
        query_rows_spent_before=spent_rows, query_rows_remaining_before=cap - spent_rows,
        probe_token_rows=0 if step == 0 else batch * length,
        signal_seconds=time.perf_counter() - started,
        signal="mean relative pre_dit input drift against per-token cached input; no attention probe")
