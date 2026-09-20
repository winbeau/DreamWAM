"""Immutable, dependency-free schedules and experiment identities."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

OPERATIONS = ("dense", "sparse", "reuse")


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def mapping(value, allowed, name, required=()):
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    unknown = set(value) - set(allowed)
    missing = set(required) - set(value)
    if unknown or missing:
        raise ValueError(f"{name}: unknown keys {sorted(unknown)}, missing keys {sorted(missing)}")
    return dict(value)


def integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True)
class Schedule:
    num_steps: int
    operations: tuple[str, ...]
    source: str = "explicit"
    refresh_every: int | None = None
    refresh_operation: str | None = None
    profile_path: str | None = None
    profile_sha256: str | None = None
    profile_policy_hash: str | None = None
    # JSON keeps externally supplied dictionaries from mutating a frozen plan.
    compatibility_json: str | None = None

    def __post_init__(self):
        integer(self.num_steps, "schedule.num_steps")
        if not isinstance(self.operations, tuple) or len(self.operations) != self.num_steps:
            raise ValueError("operations must contain exactly num_steps entries")
        if any(not isinstance(op, str) or op not in OPERATIONS for op in self.operations):
            raise ValueError(f"operations must be one of {OPERATIONS}")
        if self.operations[0] != "dense":
            raise ValueError("the first operation must be dense")

    @classmethod
    def from_mapping(cls, payload):
        if not isinstance(payload, Mapping):
            raise ValueError("schedule must be a mapping")
        kind = payload.get("kind")
        if kind == "explicit":
            p = mapping(payload, ("kind", "num_steps", "operations"), "schedule",
                        ("num_steps", "operations"))
            if not isinstance(p["operations"], (list, tuple)):
                raise ValueError("operations must be a sequence")
            return cls(p["num_steps"], tuple(p["operations"]))
        if kind == "periodic":
            p = mapping(payload, ("kind", "num_steps", "refresh_every", "refresh_operation"),
                        "schedule", ("num_steps", "refresh_every"))
            n = integer(p["num_steps"], "num_steps")
            every = integer(p["refresh_every"], "refresh_every")
            op = p.get("refresh_operation", "sparse")
            if op not in ("dense", "sparse"):
                raise ValueError("refresh_operation must be dense or sparse")
            return cls(n, tuple("dense" if i == 0 else op if i % every == 0 else "reuse"
                                for i in range(n)), kind, every, op)
        if kind == "profile":
            p = mapping(payload, ("kind", "num_steps", "path", "sha256"), "schedule",
                        ("num_steps", "path", "sha256"))
            path = Path(p["path"])
            if not path.is_absolute():
                raise ValueError("profile path must be explicit and absolute")
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != p["sha256"]:
                raise ValueError("profile SHA256 mismatch")
            doc = mapping(json.loads(data), ("schema_version", "num_steps", "operations",
                          "policy_hash", "compatibility"), "profile",
                          ("schema_version", "num_steps", "operations", "policy_hash", "compatibility"))
            if type(doc["schema_version"]) is not int or doc["schema_version"] != 1:
                raise ValueError("unsupported profile schema")
            if doc["num_steps"] != p["num_steps"]:
                raise ValueError("profile num_steps mismatch")
            compatibility = mapping(doc["compatibility"],
                ("video_length", "tokens_per_frame", "scheduler_hash"), "profile.compatibility",
                ("video_length", "tokens_per_frame", "scheduler_hash"))
            for key in ("video_length", "tokens_per_frame"):
                integer(compatibility[key], key)
            for value in (doc["policy_hash"], compatibility["scheduler_hash"]):
                if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                    raise ValueError("profile hashes must be lowercase SHA256")
            if not isinstance(doc["operations"], list):
                raise ValueError("profile operations must be a list")
            return cls(p["num_steps"], tuple(doc["operations"]), kind,
                       profile_path=str(path), profile_sha256=p["sha256"],
                       profile_policy_hash=doc["policy_hash"],
                       compatibility_json=json.dumps(compatibility, sort_keys=True))
        raise ValueError("schedule.kind must be explicit, periodic or profile")

    def describe(self, *, canonical=False):
        result = dict(kind="explicit", num_steps=self.num_steps, operations=list(self.operations))
        if not canonical and self.source == "periodic":
            result = dict(kind="periodic", num_steps=self.num_steps,
                          refresh_every=self.refresh_every, refresh_operation=self.refresh_operation)
        if not canonical and self.source == "profile":
            result = dict(kind="profile", num_steps=self.num_steps,
                          path=self.profile_path, sha256=self.profile_sha256)
        return result


@dataclass(frozen=True)
class StepContext:
    step_index: int
    num_steps: int
    video_t: object = None
    action_t: object = None


@dataclass(frozen=True)
class StepDecision:
    operation: str
    read_mode: str
    requested_q_ratio: float
    requested_kv_ratio: float
    route_refresh: bool
    source: str
    reason: str


@dataclass(frozen=True)
class StepPlan:
    schedule: Schedule
    plan_hash: str
    read_mode: str
    recompute_ratio: float
    read_ratio: float

    def decision(self, context):
        if context.num_steps != self.schedule.num_steps:
            raise ValueError("actual num_steps differs from the compiled plan")
        step = integer(context.step_index, "step_index", 0)
        if step >= self.schedule.num_steps:
            raise ValueError("step_index outside plan")
        op = self.schedule.operations[step]
        return StepDecision(op, "full" if op == "dense" else self.read_mode,
                            1.0 if op == "dense" else self.recompute_ratio if op == "sparse" else 0.0,
                            1.0 if op == "dense" else self.read_ratio,
                            op != "reuse", self.schedule.source, "compiled_schedule")


def compile_plan(config, num_steps, *, compatibility=None):
    integer(num_steps, "actual num_steps")
    if num_steps != config.schedule.num_steps:
        raise ValueError("actual num_steps differs from schedule.num_steps")
    if config.schedule.source == "profile":
        if config.policy_hash != config.schedule.profile_policy_hash:
            raise ValueError("profile budgets/selector/execution policy mismatch")
        if compatibility is not None and compatibility != json.loads(config.schedule.compatibility_json):
            raise ValueError("profile layout or scheduler mismatch")
    return StepPlan(config.schedule, config.policy_hash, config.read_mode,
                    config.recompute_ratio, config.read_ratio)
