"""M1: causal, training-free budgets from RGB/proprio observation history.

These observable-change scores are hypotheses, not task-phase or safety oracles.
No attention, denoising latent, outcome, teacher action, or simulator state enters
the controller. History advances only after a successful policy prediction.
"""

from collections import deque
from dataclasses import dataclass, replace
import hashlib
import time

import numpy as np
from PIL import Image

from .hybrid.config import HybridConfig, number
from .hybrid.schedule import integer, mapping, stable_hash


FEATURE_SCALES = dict(rgb_change=0.08, rgb_innovation=0.08,
                      translation=0.05, rotation=0.30, gripper=0.02,
                      translation_innovation=0.05)


@dataclass(frozen=True)
class BudgetLevel:
    name: str
    query_ratio: float
    read_ratio: float
    chunk_extra_passes: float | None = None

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name or not self.name.isidentifier():
            raise ValueError("budget level name must be a nonempty identifier")
        number(self.query_ratio, "query_ratio", 0, 1, open_low=True)
        number(self.read_ratio, "read_ratio", 0, 1, open_low=True)
        if self.query_ratio > self.read_ratio:
            raise ValueError("query budget cannot exceed read budget")
        if self.chunk_extra_passes is not None:
            number(self.chunk_extra_passes, "chunk_extra_passes", 0)

    def describe(self):
        result = dict(name=self.name, query_ratio=float(self.query_ratio),
                      read_ratio=float(self.read_ratio))
        if self.chunk_extra_passes is not None:
            result["chunk_extra_passes"] = float(self.chunk_extra_passes)
        return result


@dataclass(frozen=True)
class ChunkBudgetConfig:
    levels: tuple = (BudgetLevel("low", 0.1, 0.25),
                     BudgetLevel("medium", 0.2, 0.5),
                     BudgetLevel("high", 0.3, 0.75))
    thresholds: tuple = (0.5, 1.5)
    scales: tuple = tuple(FEATURE_SCALES.items())
    history_size: int = 3
    thumbnail_size: int = 32
    downshift_after: int = 2
    mode: str = "adaptive"
    fixed_level: int = 1

    def __post_init__(self):
        if not isinstance(self.levels, tuple) or not 2 <= len(self.levels) <= 8:
            raise ValueError("chunk_budget requires 2 to 8 budget levels")
        if any(not isinstance(level, BudgetLevel) for level in self.levels):
            raise ValueError("levels must be validated BudgetLevel entries")
        if len({level.name for level in self.levels}) != len(self.levels):
            raise ValueError("duplicate budget level names")
        caps = [level.chunk_extra_passes for level in self.levels]
        if any(cap is None for cap in caps) and not all(cap is None for cap in caps):
            raise ValueError("all levels must declare chunk_extra_passes, or none")
        for before, after in zip(self.levels, self.levels[1:]):
            if (after.query_ratio < before.query_ratio or after.read_ratio < before.read_ratio
                    or (before.chunk_extra_passes is not None and after.chunk_extra_passes < before.chunk_extra_passes)
                    or (after.query_ratio, after.read_ratio, after.chunk_extra_passes)
                    == (before.query_ratio, before.read_ratio, before.chunk_extra_passes)):
                raise ValueError("budget levels must increase monotonically")
        if not isinstance(self.thresholds, tuple) or len(self.thresholds) != len(self.levels) - 1:
            raise ValueError("require one threshold between adjacent budget levels")
        for value in self.thresholds:
            number(value, "threshold", 0, open_low=True)
        if any(a >= b for a, b in zip(self.thresholds, self.thresholds[1:])):
            raise ValueError("thresholds must strictly increase")
        if not isinstance(self.scales, tuple) or len(self.scales) != len(FEATURE_SCALES):
            raise ValueError("invalid feature scales")
        scales = dict(self.scales)
        if set(scales) != set(FEATURE_SCALES):
            raise ValueError("feature scale keys must match observable features")
        for key, value in scales.items():
            number(value, key + " scale", 0, open_low=True)
        for name, minimum, maximum in (("history_size", 2, 32),
                                      ("thumbnail_size", 8, 128),
                                      ("downshift_after", 1, 32)):
            value = integer(getattr(self, name), name, minimum)
            if value > maximum:
                raise ValueError(name + " exceeds supported bound")
        if self.mode not in ("adaptive", "fixed"):
            raise ValueError("chunk_budget.mode must be adaptive or fixed")
        integer(self.fixed_level, "fixed_level", 0)
        if self.fixed_level >= len(self.levels):
            raise ValueError("fixed_level is outside configured levels")

    @classmethod
    def from_mapping(cls, payload):
        p = mapping(payload, ("schema_version", "levels", "thresholds", "scales", "history_size",
                              "thumbnail_size", "downshift_after", "mode", "fixed_level"), "chunk_budget")
        if type(p.get("schema_version", 1)) is not int or p.get("schema_version", 1) != 1:
            raise ValueError("unsupported chunk_budget.schema_version")
        defaults = cls()
        levels = defaults.levels
        if "levels" in p:
            if not isinstance(p["levels"], (list, tuple)):
                raise ValueError("levels must be a sequence")
            levels = tuple(BudgetLevel(**mapping(item, ("name", "query_ratio", "read_ratio", "chunk_extra_passes"),
                "budget level", ("name", "query_ratio", "read_ratio"))) for item in p["levels"])
        thresholds = p.get("thresholds", defaults.thresholds)
        if not isinstance(thresholds, (list, tuple)):
            raise ValueError("thresholds must be a sequence")
        scales = mapping(p.get("scales", FEATURE_SCALES), FEATURE_SCALES, "scales", FEATURE_SCALES)
        options = {key: p.get(key, getattr(defaults, key)) for key in
                   ("history_size", "thumbnail_size", "downshift_after", "mode", "fixed_level")}
        return cls(levels=levels, thresholds=tuple(thresholds),
                   scales=tuple((key, scales[key]) for key in FEATURE_SCALES), **options)

    def describe(self):
        return dict(schema_version=1, levels=[level.describe() for level in self.levels],
                    thresholds=list(self.thresholds), scales=dict(self.scales),
                    history_size=self.history_size, thumbnail_size=self.thumbnail_size,
                    downshift_after=self.downshift_after, mode=self.mode, fixed_level=self.fixed_level)

    @property
    def policy_hash(self):
        return stable_hash(self.describe())

    def hybrid_configs(self, base):
        if not isinstance(base, HybridConfig) or base.read_mode != "compact":
            raise ValueError("chunk_budget requires a compact hybrid_visual configuration")
        if base.schedule.source == "profile":
            raise ValueError("chunk_budget cannot change a hash-frozen profile budget")
        # Validation also enforces equal Q/KV for a structure-only reuse control.
        return tuple(replace(base, recompute_ratio=level.query_ratio, read_ratio=level.read_ratio,
                             chunk_extra_passes=(base.chunk_extra_passes if level.chunk_extra_passes is None
                                                 else level.chunk_extra_passes))
                     for level in self.levels)


def rotation_distance(before, after):
    """Geodesic SO(3) distance: equivalent axis-angle encodings stay equivalent."""
    def quaternion(vector):
        angle = np.linalg.norm(vector)
        scale = 0.5 if angle < 1e-12 else np.sin(angle / 2) / angle
        return np.concatenate(([np.cos(angle / 2)], vector * scale))
    dot = abs(float(quaternion(before) @ quaternion(after)))
    return float(2 * np.arccos(np.clip(dot, 0, 1)))


@dataclass(frozen=True)
class BudgetDecision:
    level_index: int
    diagnostics: dict
    thumbnail: np.ndarray
    state: np.ndarray
    downshift_streak: int
    generation: int
    chunk_index: int


class ObservationBudget:
    def __init__(self, config):
        self.config = config if isinstance(config, ChunkBudgetConfig) else ChunkBudgetConfig.from_mapping(config)
        self._generation = 0
        self.reset()

    def reset(self):
        self._history = deque(maxlen=self.config.history_size)
        self._level = len(self.config.levels) - 1
        self._downshift_streak = 0
        self._chunks = 0
        self._generation += 1
        self.last_stats = {}

    def propose(self, images, state):
        started = time.perf_counter()
        if set(images) != {"agentview", "wrist"}:
            raise ValueError("M1 requires exactly agentview and wrist")
        state = np.asarray(state)
        if state.shape != (8,) or state.dtype.kind not in "fiu" or not np.isfinite(state).all():
            raise ValueError("M1 requires a finite eight-dimensional robot observation")
        state = state.astype(np.float64, copy=True)
        thumbnails, summaries = [], {}
        for camera in ("agentview", "wrist"):
            array = np.asarray(images[camera])
            if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3 or min(array.shape[:2]) == 0:
                raise ValueError("M1 camera must be nonempty uint8 HWC RGB")
            thumbnails.append(np.asarray(Image.fromarray(array).resize(
                (self.config.thumbnail_size,) * 2, Image.Resampling.BOX), dtype=np.float32) / 255)
            summaries[camera] = dict(shape=list(array.shape),
                                    sha256=hashlib.sha256(array.tobytes(order="C")).hexdigest())
        current = np.stack(thumbnails)
        features = dict.fromkeys(FEATURE_SCALES, 0.0)
        def local_change(value):
            # Retain local changes without allowing a single noisy pixel to dominate.
            return float(np.quantile(np.abs(value).mean(axis=-1), 0.90))
        if self._history:
            previous, previous_state = self._history[-1]
            features.update(rgb_change=local_change(current - previous),
                translation=float(np.linalg.norm(state[:3] - previous_state[:3])),
                rotation=rotation_distance(previous_state[3:6], state[3:6]),
                gripper=float(np.linalg.norm(state[6:] - previous_state[6:])))
            if len(self._history) > 1:
                older, older_state = self._history[-2]
                features.update(rgb_innovation=local_change(current - np.clip(2 * previous - older, 0, 1)),
                    translation_innovation=float(np.linalg.norm(state[:3] - 2 * previous_state[:3] + older_state[:3])))
        normalized = {key: features[key] / scale for key, scale in self.config.scales}
        score = max(normalized.values())
        target = sum(score >= threshold for threshold in self.config.thresholds)
        streak = 0
        if self.config.mode == "fixed":
            selected, reason = self.config.fixed_level, "fixed_budget_control"
        elif not self._history:
            selected, reason = len(self.config.levels) - 1, "bootstrap_no_history"
        elif target >= self._level:
            selected = target
            reason = "observable_change_upgrade" if target > self._level else "same_band"
        else:
            streak = self._downshift_streak + 1
            if streak >= self.config.downshift_after:
                selected, reason, streak = self._level - 1, "sustained_lower_change", 0
            else:
                selected, reason = self._level, "downshift_debounce"
        level = self.config.levels[selected]
        stats = dict(schema_version=1, chunk_index=self._chunks, history_length=len(self._history),
            config_hash=self.config.policy_hash, mode=self.config.mode, level_index=selected,
            level=level.name, budget=level.describe(), proposed_level_index=target,
            reason=reason, score=score, dominant_feature=max(normalized, key=normalized.get),
            features=features, normalized_features=normalized, downshift_streak=streak,
            input_summary=dict(images=summaries, state=state.tolist(),
                state_sha256=hashlib.sha256(state.tobytes(order="C")).hexdigest(), state_hash_dtype="float64"),
            status="PROPOSED", controller_seconds=time.perf_counter() - started)
        return BudgetDecision(selected, stats, current, state, streak, self._generation, self._chunks)

    def commit(self, decision):
        if decision.generation != self._generation or decision.chunk_index != self._chunks:
            raise RuntimeError("stale or already committed M1 decision")
        self._history.append((decision.thumbnail.copy(), decision.state.copy()))
        self._level = decision.level_index
        self._downshift_streak = decision.downshift_streak
        self._chunks += 1
        self.last_stats = dict(decision.diagnostics, status="COMMITTED")
