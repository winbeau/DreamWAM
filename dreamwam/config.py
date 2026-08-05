from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .model import DreamWAMConfig


PROJECT_ROOT = Path(__file__).absolute().parents[1]


def project_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Release paths must be project-relative: {value!r}.")
    return PROJECT_ROOT / path


@dataclass(frozen=True)
class ReleasePaths:
    dataset_root: Path
    dataset_stats: Path
    cache_root: Path
    checkpoint: Path
    output_dir: Path
    libero_plus_root: Path
    libero_plus_data_root: Path


@dataclass(frozen=True)
class PretrainedInitialization:
    wan_dit_root: Path
    action_dit_checkpoint: Path
    flow_channel_init_scale: float


@dataclass(frozen=True)
class ReleaseConfig:
    name: str
    setting: str
    paths: ReleasePaths
    initialization: PretrainedInitialization
    model: DreamWAMConfig
    training: dict[str, Any]
    evaluation: dict[str, Any]
    preprocessing: dict[str, Any]


def _tuple(value: Any, name: str) -> tuple:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty YAML list.")
    return tuple(value)


def load_release_config(path: str | Path) -> ReleaseConfig:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config: {config_path}")
    payload = yaml.safe_load(config_path.read_text())
    if not isinstance(payload, dict):
        raise TypeError("Top-level config must be a mapping.")

    setting = str(payload.get("setting", "")).strip().lower()
    if setting not in {"uncond", "joint"}:
        raise ValueError("setting must be 'uncond' or 'joint'.")
    paths_payload = payload.get("paths")
    if not isinstance(paths_payload, dict):
        raise TypeError("paths must be a mapping.")
    required_paths = {
        "dataset_root",
        "dataset_stats",
        "cache_root",
        "checkpoint",
        "output_dir",
        "libero_plus_root",
        "libero_plus_data_root",
    }
    missing_paths = sorted(required_paths - paths_payload.keys())
    if missing_paths:
        raise KeyError(f"Missing path entries: {missing_paths}")
    paths = ReleasePaths(
        **{
            key: project_path(str(paths_payload[key]))
            for key in sorted(required_paths)
        }
    )

    initialization_payload = payload.get("initialization")
    if not isinstance(initialization_payload, dict):
        raise TypeError("initialization must be a mapping.")
    required_initialization = {
        "wan_dit_root",
        "action_dit_checkpoint",
        "flow_channel_init_scale",
    }
    missing_initialization = sorted(
        required_initialization - initialization_payload.keys()
    )
    if missing_initialization:
        raise KeyError(
            f"Missing initialization entries: {missing_initialization}"
        )
    flow_channel_init_scale = float(
        initialization_payload["flow_channel_init_scale"]
    )
    if flow_channel_init_scale <= 0:
        raise ValueError("initialization.flow_channel_init_scale must be positive.")
    initialization = PretrainedInitialization(
        wan_dit_root=project_path(str(initialization_payload["wan_dit_root"])),
        action_dit_checkpoint=project_path(
            str(initialization_payload["action_dit_checkpoint"])
        ),
        flow_channel_init_scale=flow_channel_init_scale,
    )

    model_payload = dict(payload.get("model") or {})
    routing = dict(payload.get("routing") or {})
    fusion = str(routing.get("fusion", "")).strip().lower()
    if fusion not in {"sequential", "parallel"}:
        raise ValueError("routing.fusion must be 'sequential' or 'parallel'.")
    local_preview = dict(routing.get("local_preview") or {})
    model_payload.update(
        setting=setting,
        routing_fusion=fusion,
        routing_order=_tuple(routing.get("order"), "routing.order"),
        dino_injection_layers=_tuple(
            routing.get("dino_layers"), "routing.dino_layers"
        ),
        depth_injection_layers=_tuple(
            routing.get("depth_layers"), "routing.depth_layers"
        ),
        local_preview_enabled=bool(local_preview.get("enabled", False)),
        local_preview_loss_weight=float(local_preview.get("loss_weight", 0.0)),
        local_preview_delta_scale=float(local_preview.get("delta_scale", 1.0)),
        local_preview_layer_weighting=str(
            local_preview.get("layer_weighting", "uniform")
        ),
        local_preview_min_layer_weight=float(
            local_preview.get("min_layer_weight", 0.25)
        ),
    )
    loss_payload = dict(payload.get("loss") or {})
    model_payload.update(
        video_loss_weight=float(loss_payload["video"]),
        action_loss_weight=float(loss_payload["action"]),
        flow_loss_weight=float(loss_payload["flow"]),
        dino_loss_weight=float(loss_payload["dino"]["initial"]),
        dino_final_loss_weight=float(loss_payload["dino"]["final"]),
        depth_loss_weight=float(loss_payload["depth"]["initial"]),
        depth_final_loss_weight=float(loss_payload["depth"]["final"]),
        gate_l1_weight=float(loss_payload["gate_l1"]),
    )
    model = DreamWAMConfig(**model_payload)
    preprocessing = dict(payload.get("preprocessing") or {})
    for key in (
        "wan_vae_checkpoint",
        "wan_text_checkpoint",
        "wan_tokenizer",
        "raft_source",
        "raft_checkpoint",
        "dino_source",
        "dino_checkpoint",
        "dino_projection",
        "depth_source",
        "depth_checkpoint",
        "depth_projection",
    ):
        if key not in preprocessing:
            raise KeyError(f"Missing preprocessing path: {key}")
        preprocessing[key] = project_path(str(preprocessing[key]))
    return ReleaseConfig(
        name=str(payload.get("name", config_path.stem)),
        setting=setting,
        paths=paths,
        initialization=initialization,
        model=model,
        training=dict(payload.get("training") or {}),
        evaluation=dict(payload.get("evaluation") or {}),
        preprocessing=preprocessing,
    )
