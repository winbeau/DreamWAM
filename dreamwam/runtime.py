from pathlib import Path

import torch

from .config import ReleaseConfig
from .model import DreamWAMJoint, DreamWAMUncond


def build_model(
    config: ReleaseConfig,
    *,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> DreamWAMJoint:
    model_class = DreamWAMUncond if config.setting == "uncond" else DreamWAMJoint
    if device is None and dtype is None:
        return model_class(config.model)
    if device is None or dtype is None:
        raise ValueError("device and dtype must be provided together.")
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device(device):
            return model_class(config.model)
    finally:
        torch.set_default_dtype(previous_dtype)


def load_model_checkpoint(
    model: torch.nn.Module,
    checkpoint: str | Path,
    *,
    strict: bool = True,
) -> None:
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    if not isinstance(payload, dict):
        raise TypeError("Checkpoint must contain a state dictionary mapping.")
    if "mot" in payload or "world_routing_modules" in payload:
        required_sections = {
            "mot",
            "world_routing_modules",
            "flow_channel_modules",
            "proprio_encoder",
        }
        missing_sections = sorted(required_sections - payload.keys())
        if missing_sections:
            raise KeyError(
                f"DreamWAM checkpoint is missing sections: {missing_sections}"
            )
        non_mappings = sorted(
            key for key in required_sections if not isinstance(payload[key], dict)
        )
        if non_mappings:
            raise TypeError(
                f"DreamWAM checkpoint sections must be mappings: {non_mappings}"
            )

        flow_snapshot_map = {
            "0.weight": "mixtures.video.patch_embedding.weight",
            "0.bias": "mixtures.video.patch_embedding.bias",
            "1.modulation": "mixtures.video.head.modulation",
            "1.head.weight": "mixtures.video.head.head.weight",
            "1.head.bias": "mixtures.video.head.head.bias",
        }
        if set(payload["flow_channel_modules"]) != set(flow_snapshot_map):
            raise KeyError(
                "flow_channel_modules does not match the released channel-concat layout."
            )
        for flow_key, mot_key in flow_snapshot_map.items():
            if mot_key not in payload["mot"]:
                raise KeyError(f"Checkpoint MoT is missing {mot_key!r}.")
            if not torch.equal(
                payload["flow_channel_modules"][flow_key],
                payload["mot"][mot_key],
            ):
                raise ValueError(
                    f"Checkpoint flow snapshot differs from MoT parameter {mot_key!r}."
                )

        state = {}
        for key, value in payload["mot"].items():
            mapped = "mot." + key
            mapped = mapped.replace("mixtures.video", "video_expert")
            mapped = mapped.replace("mixtures.action", "action_expert")
            mapped = mapped.replace(
                "action_expert.action_encoder", "action_expert.action_embedding"
            )
            mapped = mapped.replace("video_expert.head.head", "video_expert.head.proj")
            state[mapped] = value
        for key, value in payload["world_routing_modules"].items():
            mapped = "world_residual." + key
            mapped = mapped.replace("da3d8", "depth")
            mapped = mapped.replace("heads.dino.head", "heads.dino.proj")
            mapped = mapped.replace("heads.depth.head", "heads.depth.proj")
            mapped = mapped.replace("adapters", "residual_branches")
            state[mapped] = value
        for key, value in payload["proprio_encoder"].items():
            state[f"proprio_encoder.{key}"] = value
        payload = state
    for key in ("model", "state_dict", "module"):
        candidate = payload.get(key)
        if isinstance(candidate, dict):
            payload = candidate
            break
    model.load_state_dict(payload, strict=strict)
