"""action-eval adapter for DreamWAM.

This is the only file DreamWAM needs in order to be evaluated by action-eval. It is
deliberately thin: it wires the platform's observation and action contract onto the
model's existing ``build_policy``/``predict_action`` and adds no modelling code of
its own.

Where each responsibility lives:

===========================================  ==========================
concern                                      owner
===========================================  ==========================
image flip                                   action-eval ``libero/backend.py``
center-crop, resize, camera concatenation    DreamWAM ``policy.py``
prompt template, proprio/action normalisation DreamWAM ``policy.py``
gripper sign and binarisation                DreamWAM ``policy.py``
which GPU, timeouts, episode loop, success   action-eval
===========================================  ==========================

The flip belongs to the benchmark because it owns the observation profile, so this
adapter must *not* flip again: ``action_eval/benchmarks/libero/backend.py`` already applies the same
``[::-1, ::-1]`` that DreamWAM's own ``evaluation/rollout.py`` applies, and doing it a
second time would silently feed the model an upside-down view. The center-crop/resize
and the ``agentview|wrist`` concatenation are the model's own and happen inside
``DreamWAMPolicy.predict_action``; the adapter does not repeat them either.

The runner injects three fully resolved paths into the adapter options, so this file
never has to guess how a relative path should be interpreted:

``model_root``     the fork checkout (``policy.repo_root``)
``model_config``   the DreamWAM release YAML (``policy.model_config``)
``checkpoint``     the checkpoint path (``policy.checkpoint``)

The injected ``checkpoint`` is authoritative. DreamWAM's release YAML also carries its
own ``paths.checkpoint``; when both exist and disagree the injected path is loaded and
the disagreement is recorded in the fingerprint, so the report can never claim a
checkpoint that did not run.

Options (``policy.options`` in the experiment YAML), all optional:
``action_horizon`` number of actions returned per chunk. Must be >= the protocol's
                   ``replan_steps`` or the platform rejects the chunk.
``denoising_steps`` sampling steps; recorded because it changes the model's output
                   and therefore has to be part of the comparison.
``rng_mode``       ``fixed_per_predict`` (default for DreamWAM, whose
                   ``sample_action`` receives a fixed seed from its config) or
                   ``episode_stream``. Declared honestly because it changes what a
                   repeated episode means.
``hash_checkpoint`` default ``True``. Set ``false`` to skip the SHA-256 of the
                   checkpoint when a run is already covered by a recorded hash.
``prompt_cache``   exact frozen text-encoding reuse, e.g. ``{capacity: 8}``.
                   Opt-in on both Dense and Sparse; cleared at each episode reset.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import time
from pathlib import Path

import numpy as np

from action_eval_sdk import (
    ACTION_PROFILE_LIBERO_OSC_POSE_V1,
    OBSERVATION_PROFILE_LIBERO_2CAM_V1,
    RNG_MODE_EPISODE_STREAM,
    RNG_MODE_FIXED_PER_PREDICT,
    PolicyDescription,
    Prediction,
)

#: Imported lazily inside the factory so that merely importing this file (for a
#: contract check) does not require torch or the model package to be importable.
_DREAMWAM = {}


def _import_dreamwam(model_root: Path):
    """Import DreamWAM's public entry points from its own checkout."""
    if _DREAMWAM:
        return _DREAMWAM
    import sys

    root = str(model_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    from dreamwam.config import load_release_config  # type: ignore
    from dreamwam.policy import build_policy  # type: ignore
    from dreamwam.sparse.config import config_hash  # type: ignore

    _DREAMWAM["load_release_config"] = load_release_config
    _DREAMWAM["build_policy"] = build_policy
    _DREAMWAM["config_hash"] = config_hash
    return _DREAMWAM


def _sha256(path: Path, *, chunk: int = 1 << 22) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while True:
                block = handle.read(chunk)
                if not block:
                    break
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


#: ``describe()`` is called once per worker at startup, but hashing a 12 GiB checkpoint
#: costs real time, so the digest is cached per (path, size, mtime).
_CHECKPOINT_DIGESTS: dict[tuple[str, int, int], str | None] = {}


def _cached_sha256(path: Path) -> str | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in _CHECKPOINT_DIGESTS:
        _CHECKPOINT_DIGESTS[key] = _sha256(path)
    return _CHECKPOINT_DIGESTS[key]


class DreamWAMPolicy:
    def __init__(self, config: dict, device: str):
        self.config = dict(config)
        self.device = device
        #: ``policy.options`` from the experiment YAML; read throughout this constructor.
        options = self.config
        self.model_root = Path(self.config.get("model_root") or os.getcwd()).expanduser()

        model_config_path = self.config.get("model_config")
        if not model_config_path:
            raise ValueError(
                "DreamWAM adapter requires `model_config`: the path to a DreamWAM "
                "release YAML (e.g. configs/dreamwam_joint.yaml)."
            )
        model_config_path = Path(model_config_path).expanduser()
        if not model_config_path.is_absolute():
            model_config_path = self.model_root / model_config_path
        if not model_config_path.is_file():
            raise FileNotFoundError(f"DreamWAM release config not found: {model_config_path}")

        dreamwam = _import_dreamwam(self.model_root)
        load_release_config = dreamwam["load_release_config"]
        build_policy = dreamwam["build_policy"]
        config_hash = dreamwam["config_hash"]

        release = load_release_config(model_config_path)
        self.release_config_path = model_config_path

        # The injected path is authoritative; the release YAML's own path is only a
        # default. Loading the YAML path while reporting the injected one would make
        # the fingerprint describe a checkpoint that never ran, so the injected path
        # is carried into the ReleaseConfig that build_policy actually consumes.
        declared_checkpoint = Path(release.paths.checkpoint)
        injected_checkpoint = self.config.get("checkpoint")
        if injected_checkpoint:
            candidate = Path(injected_checkpoint).expanduser()
            if not candidate.is_absolute():
                candidate = self.model_root / candidate
            if not candidate.is_file():
                raise FileNotFoundError(
                    f"policy.checkpoint does not exist: {candidate}"
                )
            self.checkpoint_path = candidate
            self._checkpoint_source = "policy.checkpoint"
            self._checkpoint_conflict = (
                None
                if candidate.resolve() == declared_checkpoint.resolve()
                else {
                    "model_config_checkpoint": str(declared_checkpoint),
                    "policy_checkpoint": str(candidate),
                }
            )
            release = dataclasses.replace(
                release,
                paths=dataclasses.replace(release.paths, checkpoint=candidate),
            )
        else:
            self.checkpoint_path = declared_checkpoint
            self._checkpoint_source = "model_config"
            self._checkpoint_conflict = None

        # torch is imported here rather than at module import so a contract test can
        # inspect this file without a GPU environment.
        import torch  # type: ignore

        self._torch = torch
        if device.startswith("cuda") and not torch.cuda.is_available():
            # Never fall back to CPU silently: a CPU rollout is orders of magnitude
            # slower, changes the numerics and would burn the whole evaluation budget
            # while still producing a success rate that looks legitimate.
            raise RuntimeError(
                "DreamWAM adapter was given a CUDA device but torch reports that no "
                "CUDA device is available; refusing to fall back to CPU. Check that "
                "CUDA_VISIBLE_DEVICES selects a working GPU."
            )
        self.device = device
        # Sparse-WAM options are parsed and validated by the model itself; passing the raw
        # mapping keeps the adapter free of modelling decisions, and an unknown key fails
        # at startup instead of silently running dense.
        self._sparse_options = options.get("sparse")
        self.policy = build_policy(release, device=device, sparse=self._sparse_options,
                                   visual_cache=options.get("visual_cache"),
                                   prompt_cache=options.get("prompt_cache"),
                                   fresh_visual_tokens=options.get("fresh_visual_tokens"),
                                   hybrid_visual=options.get("hybrid_visual"),
                                   chunk_budget=options.get("chunk_budget"))
        self._sparse_hash = config_hash(self.policy.sparse_config)

        # ``build_policy`` gives the policy its own reference to the release YAML's
        # ``evaluation`` dict. Reading and writing that same dict is what makes the
        # options below change the model instead of only changing the report.
        self._evaluation = self.policy.evaluation
        if options.get("action_horizon") is not None:
            self._evaluation["action_horizon"] = int(options["action_horizon"])
        if options.get("denoising_steps") is not None:
            self._evaluation["denoising_steps"] = int(options["denoising_steps"])
        for key in ("action_horizon", "video_frames", "denoising_steps"):
            if int(self._evaluation[key]) <= 0:
                raise ValueError(f"evaluation.{key} must be positive")
        if getattr(self.policy, "_hybrid_visual_runtime", None) is not None:
            try:
                self.policy.validate_inference_options()
            except BaseException:
                self.policy.close()
                raise

        self._action_horizon = int(self._evaluation["action_horizon"])
        self._rng_mode = str(options.get("rng_mode") or RNG_MODE_FIXED_PER_PREDICT)
        if self._rng_mode not in {RNG_MODE_FIXED_PER_PREDICT, RNG_MODE_EPISODE_STREAM}:
            raise ValueError(
                f"rng_mode must be {RNG_MODE_FIXED_PER_PREDICT!r} or "
                f"{RNG_MODE_EPISODE_STREAM!r}, got {self._rng_mode!r}"
            )
        self._episode_seed = int(self._evaluation.get("seed", 0))
        self._hash_checkpoint = bool(options.get("hash_checkpoint", True))
        self.predict_calls = 0
        self._fingerprint: dict[str, object] = {}

    # -- contract ----------------------------------------------------------
    def describe(self) -> PolicyDescription:
        evaluation = self._evaluation
        self._fingerprint = {
            "adapter": "dreamwam",
            "release_config": str(self.release_config_path),
            "release_config_sha256": _sha256(self.release_config_path),
            "checkpoint": str(self.checkpoint_path),
            "checkpoint_source": self._checkpoint_source,
            "checkpoint_size_bytes": (
                self.checkpoint_path.stat().st_size if self.checkpoint_path.is_file() else None
            ),
            "checkpoint_sha256": (
                _cached_sha256(self.checkpoint_path)
                if self._hash_checkpoint
                else None
            ),
            "checkpoint_conflict": self._checkpoint_conflict,
            "setting": getattr(self.policy.config, "setting", None),
            "action_horizon": self._action_horizon,
            "denoising_steps": int(evaluation.get("denoising_steps", 0)),
            "video_frames": int(evaluation.get("video_frames", 0)),
            "replan_steps": int(evaluation.get("replan_steps", 0)),
            "wait_steps": int(evaluation.get("wait_steps", 0)),
            "binarize_gripper": bool(evaluation.get("binarize_gripper", False)),
            "model_seed": self._episode_seed,
            "rng_mode": self._rng_mode,
            "sparse": self.policy.sparse_config.describe(),
            "sparse_config_hash": self._sparse_hash,
            "visual_cache": self.policy.visual_cache_config,
        }
        if self.policy.prompt_cache_config is not None:
            self._fingerprint["prompt_cache"] = self.policy.prompt_cache_config
        if self.policy.fresh_visual_config is not None:
            self._fingerprint["fresh_visual_tokens"] = self.policy.fresh_visual_config
            self._fingerprint["fresh_visual_semantics"] = "per-step frame-quota selection; current-input bypass; no visual cache"
        if getattr(self.policy, "hybrid_visual_config", None) is not None:
            runtime = self.policy._hybrid_visual_runtime
            self._fingerprint["hybrid_visual"] = self.policy.hybrid_visual_config
            self._fingerprint["hybrid_plan_hash"] = runtime.base_config.policy_hash
            if runtime.base_config.step_router is None:
                self._fingerprint["hybrid_operations"] = list(runtime.base_config.schedule.operations)
            else:
                self._fingerprint["hybrid_operations"] = "adaptive; actual operations in per-prediction diagnostics"
                self._fingerprint["hybrid_reference_schedule"] = list(runtime.base_config.schedule.operations)
        if getattr(self.policy, "chunk_budget_config", None) is not None:
            self._fingerprint["chunk_budget"] = self.policy.chunk_budget_config
            self._fingerprint["chunk_budget_hash"] = self.policy._chunk_budget_runtime.config.policy_hash
            self._fingerprint["chunk_budget_semantics"] = (
                "causal RGB/proprio history; budget before denoising; reset per episode; "
                "observable-change proxy, not validated task-phase recognition")
        notes = (
            "DreamWAM released checkpoint through its own build_policy; the benchmark "
            "flips the images and DreamWAM center-crops, resizes and concatenates them "
            "inside predict_action, so this adapter applies no preprocessing of its own"
        )
        if self.policy.sparse_config.enabled:
            notes += (
                "; Sparse-WAM is enabled: only the VV key set is restricted, while A->V "
                "and A->A stay dense and jointly normalised, so a run is comparable with "
                "dense only through its measured density"
            )
        if self._rng_mode == RNG_MODE_FIXED_PER_PREDICT:
            notes += (
                "; note that the model seeds every sampling call from its own config, so "
                "repeats of one initial state are not independent samples"
            )
        return PolicyDescription(
            observation_profile=OBSERVATION_PROFILE_LIBERO_2CAM_V1,
            action_profile=ACTION_PROFILE_LIBERO_OSC_POSE_V1,
            rng_mode=self._rng_mode,
            action_horizon=self._action_horizon,
            fingerprint=self._fingerprint,
            notes=notes,
        )

    def reset(self, episode: dict) -> None:
        """Per-episode reset.

        Visual and action state is local to each prediction. The optional exact
        instruction cache is cleared here, so every episode pays its first encoding.
        Prediction counters are reported per episode; cache counters span the worker.
        """
        self.policy.reset()
        self.predict_calls = 0
        self._episode_id = episode.get("episode_id")
        if self._rng_mode == RNG_MODE_EPISODE_STREAM:
            # Derive a per-episode seed from the protocol seed and the episode id so
            # repeats genuinely differ. Only used when the adapter is told to.
            import zlib

            mixed = zlib.crc32(str(self._episode_id).encode()) ^ self._episode_seed
            # ``self._evaluation`` is the policy's own evaluation dict, so writing the
            # seed here is what makes the next sample_action call use it.
            self._evaluation["seed"] = mixed % (2**31 - 1)

    def predict(self, observation: dict) -> Prediction:
        images = observation.get("images") or {}
        state = np.asarray(observation.get("state"))
        instruction = observation.get("instruction")
        if not isinstance(instruction, str) or not instruction:
            raise ValueError("observation is missing a task instruction")
        if state.shape != (8,):
            raise ValueError(f"expected an 8-dim proprio vector, got {state.shape}")
        if not np.isfinite(state).all():
            raise ValueError("proprio vector contains non-finite values")
        expected_cameras = {"agentview", "wrist"}
        if set(images) != expected_cameras:
            raise ValueError(
                f"expected cameras {sorted(expected_cameras)}, got {sorted(images)}"
            )
        for name, image in images.items():
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(
                    f"camera {name} is {image.dtype} with shape {image.shape}; the "
                    "observation profile guarantees uint8 [H,W,3]"
                )

        started = time.monotonic()
        actions = self.policy.predict_action(
            images={"agentview": images["agentview"], "wrist": images["wrist"]},
            state=state.astype(np.float32),
            instruction=instruction,
        )
        elapsed = time.monotonic() - started
        actions = np.asarray(actions, dtype=np.float32)
        if not np.isfinite(actions).all():
            raise ValueError("DreamWAM produced non-finite actions")
        self.predict_calls += 1
        diagnostics = {
            "horizon": int(actions.shape[0]),
            "denoising_steps": int(self._evaluation.get("denoising_steps", 0)),
            "predict_call_index": self.predict_calls,
            "sparse_config_hash": self._sparse_hash,
        }
        visual_cache = self.policy._visual_cache_runtime
        if visual_cache is not None:
            diagnostics["visual_cache"] = dict(visual_cache.last_stats)
        if self.policy._prompt_cache_runtime is not None:
            diagnostics["prompt_cache"] = self.policy._prompt_cache_runtime.stats()
        if self.policy._fresh_visual_runtime is not None:
            diagnostics["fresh_visual_tokens"] = dict(self.policy._fresh_visual_runtime.last_stats)
        if getattr(self.policy, "_hybrid_visual_runtime", None) is not None:
            diagnostics["hybrid_visual"] = dict(self.policy._hybrid_visual_runtime.last_stats)
        if getattr(self.policy, "_chunk_budget_runtime", None) is not None:
            diagnostics["chunk_budget"] = dict(self.policy._chunk_budget_runtime.last_stats)
        if self.policy.sparse_config.enabled:
            # Executed density, not the requested one: a budget can be clamped by the
            # structural floor or replaced by a fallback route.
            counters = self.policy.model.mot.sparse_stats()
            diagnostics.update(
                {
                    "sparse_density": counters.get("mean_density"),
                    "sparse_fallback_fraction": counters.get("mean_fallback_fraction"),
                    "sparse_layer_calls": counters.get("calls"),
                }
            )
        return Prediction(
            actions=actions,
            timing={"predict_seconds": round(elapsed, 6)},
            diagnostics=diagnostics,
        )

    def close(self) -> None:
        policy = getattr(self, "policy", None)
        if policy is not None:
            policy.close()
            # Drop the model and free the cache so a long run does not accumulate
            # fragmentation across episodes.
            for attribute in ("model", "vae", "text_encoder"):
                if hasattr(policy, attribute):
                    setattr(policy, attribute, None)
        self.policy = None
        torch = getattr(self, "_torch", None)
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()


def create_policy(config: dict, device: str):
    return DreamWAMPolicy(config, device)
