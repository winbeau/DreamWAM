import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


class CachedDreamWAMDataset(Dataset):
    """Training samples produced by scripts/precompute_cache.py."""

    def __init__(
        self,
        cache_root: str | Path,
        *,
        model_config: Any,
        action_horizon: int,
        video_frames: int,
    ):
        if action_horizon <= 0 or video_frames <= 1:
            raise ValueError("action_horizon and video_frames must be positive.")
        self.cache_root = Path(cache_root)
        self.model_config = model_config
        self.action_horizon = int(action_horizon)
        self.video_frames = int(video_frames)
        index_path = self.cache_root / "index.jsonl"
        if not index_path.is_file():
            raise FileNotFoundError(f"Missing cache index: {index_path}")
        self.entries = [
            json.loads(line)
            for line in index_path.read_text().splitlines()
            if line.strip()
        ]
        if not self.entries:
            raise ValueError(f"Cache index is empty: {index_path}")
        contexts_path = self.cache_root / "contexts.pt"
        if not contexts_path.is_file():
            raise FileNotFoundError(f"Missing cached text contexts: {contexts_path}")
        self.contexts = torch.load(
            contexts_path,
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(self.contexts, list) or not self.contexts:
            raise TypeError(f"Cached text contexts must be a non-empty list.")
        for context_id, context in enumerate(self.contexts):
            if not isinstance(context, dict) or set(context) != {
                "context",
                "context_mask",
            }:
                raise TypeError(f"Invalid cached context at index {context_id}.")
            if not all(isinstance(value, torch.Tensor) for value in context.values()):
                raise TypeError(f"Cached context {context_id} contains a non-tensor value.")
            if context["context"].ndim != 2 or context["context_mask"].shape != (
                context["context"].shape[0],
            ):
                raise ValueError(f"Invalid cached context shapes at index {context_id}.")
            if context["context"].shape[1] != self.model_config.text_dim:
                raise ValueError(
                    f"Cached context width must be {self.model_config.text_dim}, "
                    f"got {context['context'].shape[1]} at index {context_id}."
                )
            if not context["context"].is_floating_point():
                raise TypeError(f"Cached context {context_id} must be floating point.")
            if context["context_mask"].dtype != torch.bool:
                raise TypeError(f"Cached context mask {context_id} must be bool.")

    def _validate_sample(self, sample: dict, sample_path: Path) -> None:
        expected_keys = {
            "video_latents",
            "flow_latents",
            "dino_latents",
            "depth_latents",
            "action",
            "action_is_pad",
            "proprio",
            "image_is_pad",
        }
        if set(sample) != expected_keys:
            missing = sorted(expected_keys - sample.keys())
            extra = sorted(sample.keys() - expected_keys)
            raise KeyError(
                f"Invalid cached sample fields in {sample_path}; "
                f"missing={missing}, extra={extra}."
            )
        if not all(isinstance(value, torch.Tensor) for value in sample.values()):
            raise TypeError(f"Cached sample contains a non-tensor value: {sample_path}")

        video = sample["video_latents"]
        expected_video_channels = self.model_config.video_latent_dim
        if video.ndim != 4 or video.shape[0] != expected_video_channels:
            raise ValueError(
                f"video_latents must be [C,T,H,W] with C={expected_video_channels}, "
                f"got {tuple(video.shape)} in {sample_path}."
            )
        expected_world_shapes = {
            "flow_latents": (self.model_config.flow_latent_dim, *video.shape[1:]),
            "dino_latents": (self.model_config.dino_latent_dim, *video.shape[1:]),
            "depth_latents": (self.model_config.depth_latent_dim, *video.shape[1:]),
        }
        for key, expected_shape in expected_world_shapes.items():
            if tuple(sample[key].shape) != expected_shape:
                raise ValueError(
                    f"{key} must have shape {expected_shape}, "
                    f"got {tuple(sample[key].shape)} in {sample_path}."
                )

        expected_shapes = {
            "action": (self.action_horizon, self.model_config.action_dim),
            "action_is_pad": (self.action_horizon,),
            "proprio": (self.action_horizon, self.model_config.proprio_dim),
            "image_is_pad": (self.video_frames,),
        }
        for key, expected_shape in expected_shapes.items():
            if tuple(sample[key].shape) != expected_shape:
                raise ValueError(
                    f"{key} must have shape {expected_shape}, "
                    f"got {tuple(sample[key].shape)} in {sample_path}."
                )
        for key in expected_keys - {"action_is_pad", "image_is_pad"}:
            if not sample[key].is_floating_point():
                raise TypeError(f"{key} must be floating point in {sample_path}.")
        for key in ("action_is_pad", "image_is_pad"):
            if sample[key].dtype != torch.bool:
                raise TypeError(f"{key} must be bool in {sample_path}.")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        entry = self.entries[index]
        relative_path = Path(entry["path"])
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"Unsafe cache entry path: {relative_path}")
        sample_path = self.cache_root / relative_path
        if not sample_path.is_file():
            raise FileNotFoundError(f"Missing cached sample: {sample_path}")
        sample = torch.load(sample_path, map_location="cpu", weights_only=True)
        if not isinstance(sample, dict):
            raise TypeError(f"Cached sample must be a dict: {sample_path}")
        self._validate_sample(sample, sample_path)
        context_id = entry.get("context_id")
        if not isinstance(context_id, int) or not 0 <= context_id < len(self.contexts):
            raise ValueError(f"Invalid context_id in cache entry: {entry}")
        if "context" in sample or "context_mask" in sample:
            raise ValueError(f"Cached sample duplicates shared text context: {sample_path}")
        sample.update(self.contexts[context_id])
        return sample
