from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterator

import torch
from safetensors import safe_open
from torch.nn import functional as F

from .config import PretrainedInitialization
from .experts import ActionDiT
from .model import DreamWAMConfig, DreamWAMJoint


WAN_DIT_INDEX = "diffusion_pytorch_model.safetensors.index.json"
ACTION_BACKBONE_SKIP_PREFIXES = ("action_embedding.", "head.")
FASTWAM_ACTION_PAYLOAD_SKIP_PREFIXES = ("action_encoder.", "head.")


@dataclass(frozen=True)
class InitializationReport:
    video_tensors: int
    action_backbone_tensors: int
    flow_channel_init_scale: float


@dataclass(frozen=True)
class ActionBackboneReport:
    copied_tensors: int
    interpolated_tensors: int


def validate_pretrained_initialization_files(
    initialization: PretrainedInitialization,
) -> None:
    _load_wan_weight_map(initialization.wan_dit_root)
    if not initialization.action_dit_checkpoint.is_file():
        raise FileNotFoundError(
            "Missing FastWAM ActionDiT backbone: "
            f"{initialization.action_dit_checkpoint}"
        )


def _load_wan_weight_map(wan_root: Path) -> dict[str, str]:
    index_path = wan_root / WAN_DIT_INDEX
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing Wan VideoDiT index: {index_path}")
    payload = json.loads(index_path.read_text())
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise TypeError(f"Wan VideoDiT index has no weight_map: {index_path}")
    if not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in weight_map.items()
    ):
        raise TypeError(f"Wan VideoDiT weight_map is malformed: {index_path}")
    for shard_name in set(weight_map.values()):
        shard_path = Path(shard_name)
        if shard_path.name != shard_name or shard_path.suffix != ".safetensors":
            raise ValueError(f"Unexpected Wan shard path in {index_path}: {shard_name}")
        if not (wan_root / shard_name).is_file():
            raise FileNotFoundError(f"Missing Wan VideoDiT shard: {wan_root / shard_name}")
    return weight_map


def _iter_wan_tensors(
    wan_root: Path,
    weight_map: dict[str, str],
    selected_keys: set[str],
) -> Iterator[tuple[str, torch.Tensor]]:
    missing = sorted(selected_keys - weight_map.keys())
    if missing:
        raise KeyError(f"Wan VideoDiT is missing tensors: {missing[:10]}")

    indexed_by_shard: dict[str, set[str]] = defaultdict(set)
    selected_by_shard: dict[str, list[str]] = defaultdict(list)
    for key, shard_name in weight_map.items():
        indexed_by_shard[shard_name].add(key)
        if key in selected_keys:
            selected_by_shard[shard_name].append(key)

    for shard_name in sorted(selected_by_shard):
        shard_path = wan_root / shard_name
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            actual_keys = set(handle.keys())
            expected_keys = indexed_by_shard[shard_name]
            if actual_keys != expected_keys:
                raise KeyError(
                    f"Wan shard/index mismatch for {shard_path}: "
                    f"missing={sorted(expected_keys - actual_keys)[:10]}, "
                    f"unexpected={sorted(actual_keys - expected_keys)[:10]}."
                )
            for key in sorted(selected_by_shard[shard_name]):
                yield key, handle.get_tensor(key)


def _copy_tensor(target: torch.Tensor, source: torch.Tensor, name: str) -> None:
    if target.is_meta:
        raise ValueError(f"Cannot initialize meta tensor: {name}")
    if tuple(target.shape) != tuple(source.shape):
        raise ValueError(
            f"Shape mismatch for {name}: expected {tuple(target.shape)}, "
            f"got {tuple(source.shape)}."
        )
    target.copy_(source.to(device=target.device, dtype=target.dtype))


def _load_wan_video_dit(
    model: DreamWAMJoint,
    initialization: PretrainedInitialization,
) -> int:
    video_state = model.video_expert.state_dict()
    source_to_target = {
        target_key.replace("head.proj", "head.head"): target_key
        for target_key in video_state
    }
    weight_map = _load_wan_weight_map(initialization.wan_dit_root)
    expected_source_keys = set(source_to_target)
    if set(weight_map) != expected_source_keys:
        raise KeyError(
            "Wan VideoDiT key set does not match DreamWAM VideoDiT: "
            f"missing={sorted(expected_source_keys - weight_map.keys())[:10]}, "
            f"unexpected={sorted(weight_map.keys() - expected_source_keys)[:10]}."
        )

    video_channels = int(model.config.video_latent_dim)
    flow_channels = int(model.config.flow_latent_dim)
    scale = float(initialization.flow_channel_init_scale)
    loaded = 0
    with torch.no_grad():
        for source_key, source in _iter_wan_tensors(
            initialization.wan_dit_root,
            weight_map,
            expected_source_keys,
        ):
            target_key = source_to_target[source_key]
            target = video_state[target_key]
            if source_key == "patch_embedding.weight":
                expected_shape = (
                    source.shape[0],
                    video_channels + flow_channels,
                    *source.shape[2:],
                )
                if (
                    source.shape[1] != video_channels
                    or tuple(target.shape) != expected_shape
                ):
                    raise ValueError(
                        "Wan patch embedding cannot initialize RGB/Flow channel concat: "
                        f"source={tuple(source.shape)}, target={tuple(target.shape)}."
                    )
                _copy_tensor(
                    target[:, :video_channels],
                    source,
                    "video_expert.patch_embedding.weight[:, :video_channels]",
                )
                source_std = (
                    source.to(dtype=target.dtype).float().std().clamp_min(1.0e-6).item()
                )
                torch.nn.init.normal_(
                    target[:, video_channels:],
                    mean=0.0,
                    std=source_std * scale,
                )
            elif source_key == "head.head.weight":
                video_outputs = int(source.shape[0])
                expected_outputs = video_outputs + flow_channels * int(
                    target.shape[0] // (video_channels + flow_channels)
                )
                if (
                    target.shape[0] != expected_outputs
                    or target.shape[1:] != source.shape[1:]
                ):
                    raise ValueError(
                        "Wan output head cannot initialize RGB/Flow channel concat: "
                        f"source={tuple(source.shape)}, target={tuple(target.shape)}."
                    )
                _copy_tensor(
                    target[:video_outputs],
                    source,
                    "video_expert.head.proj.weight[:video_outputs]",
                )
                source_std = (
                    source.to(dtype=target.dtype).float().std().clamp_min(1.0e-6).item()
                )
                torch.nn.init.normal_(
                    target[video_outputs:],
                    mean=0.0,
                    std=source_std * scale,
                )
            elif source_key == "head.head.bias":
                video_outputs = int(source.shape[0])
                if target.ndim != 1 or target.shape[0] <= video_outputs:
                    raise ValueError(
                        "Wan output bias cannot initialize RGB/Flow channel concat: "
                        f"source={tuple(source.shape)}, target={tuple(target.shape)}."
                    )
                _copy_tensor(
                    target[:video_outputs],
                    source,
                    "video_expert.head.proj.bias[:video_outputs]",
                )
                torch.nn.init.zeros_(target[video_outputs:])
            else:
                _copy_tensor(target, source, f"video_expert.{target_key}")
            loaded += 1
    return loaded


def _load_action_dit_backbone(
    model: DreamWAMJoint,
    checkpoint: Path,
) -> int:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing FastWAM ActionDiT backbone: {checkpoint}")
    payload = torch.load(
        checkpoint,
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    if not isinstance(payload, dict):
        raise TypeError(f"ActionDiT checkpoint must be a mapping: {checkpoint}")
    policy = payload.get("policy")
    expected_policy = {
        "skip_prefixes": list(FASTWAM_ACTION_PAYLOAD_SKIP_PREFIXES),
        "alpha_scaling": True,
        "interpolation": "sequential_1d_linear_align_corners_true",
    }
    if policy != expected_policy:
        raise ValueError(
            f"ActionDiT checkpoint uses an unexpected preprocessing policy: {policy}"
        )

    config = model.config
    expected_meta: dict[str, int | float] = {
        "hidden_dim": int(config.action_hidden_dim),
        "ffn_dim": int(config.action_ffn_dim),
        "num_layers": int(config.num_layers),
        "num_heads": int(config.num_heads),
        "attn_head_dim": int(config.attn_head_dim),
        "text_dim": int(config.text_dim),
        "freq_dim": int(config.freq_dim),
        "eps": float(config.eps),
    }
    if payload.get("meta") != expected_meta:
        raise ValueError(
            "ActionDiT checkpoint metadata does not match the release model: "
            f"expected={expected_meta}, got={payload.get('meta')}."
        )

    backbone = payload.get("backbone_state_dict")
    if not isinstance(backbone, dict) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in backbone.items()
    ):
        raise TypeError(f"Invalid ActionDiT backbone state: {checkpoint}")
    action_state = model.action_expert.state_dict()
    expected_keys = {
        key
        for key in action_state
        if not key.startswith(ACTION_BACKBONE_SKIP_PREFIXES)
    }
    if set(backbone) != expected_keys:
        raise KeyError(
            "ActionDiT backbone key mismatch: "
            f"missing={sorted(expected_keys - backbone.keys())[:10]}, "
            f"unexpected={sorted(backbone.keys() - expected_keys)[:10]}."
        )

    with torch.no_grad():
        for key in sorted(expected_keys):
            _copy_tensor(action_state[key], backbone[key], f"action_expert.{key}")
    return len(expected_keys)


def _verify_new_module_initialization(model: DreamWAMJoint) -> None:
    for name in model.world_residual.stream_names:
        gate = model.world_residual.gates[name]
        output = model.world_residual.residual_branches[name][-1]
        if torch.count_nonzero(gate.weight).item() != 0:
            raise RuntimeError(f"Residual gate {name} must start with zero weights.")
        expected_bias = torch.full_like(gate.bias, -4.0)
        if not torch.equal(gate.bias, expected_bias):
            raise RuntimeError(f"Residual gate {name} must start with bias -4.")
        if torch.count_nonzero(output.weight).item() != 0 or torch.count_nonzero(
            output.bias
        ).item() != 0:
            raise RuntimeError(
                f"Residual branch {name} must start as an identity-preserving zero delta."
            )


def initialize_from_fastwam_pretrained(
    model: DreamWAMJoint,
    initialization: PretrainedInitialization,
) -> InitializationReport:
    """Apply the same pretrained component initialization used by FastWAM.

    Wan2.2 initializes VideoDiT, including an RGB-preserving Flow channel
    expansion. The preprocessed FastWAM payload initializes only the ActionDiT
    backbone. Action input/output layers, proprio encoding, and DreamWAM routing
    modules remain newly initialized by their constructors. The frozen Wan VAE
    and UMT5 components are strict-loaded separately by ``dreamwam.components``.
    """

    video_tensors = _load_wan_video_dit(model, initialization)
    action_tensors = _load_action_dit_backbone(
        model,
        initialization.action_dit_checkpoint,
    )
    _verify_new_module_initialization(model)
    return InitializationReport(
        video_tensors=video_tensors,
        action_backbone_tensors=action_tensors,
        flow_channel_init_scale=float(initialization.flow_channel_init_scale),
    )


def _interpolate_last_dim(tensor: torch.Tensor, new_size: int) -> torch.Tensor:
    if tensor.shape[-1] == new_size:
        return tensor
    flat = tensor.reshape(-1, 1, tensor.shape[-1]).float()
    flat = F.interpolate(flat, size=new_size, mode="linear", align_corners=True)
    return flat.reshape(*tensor.shape[:-1], new_size)


def _resize_tensor_to_shape(
    source: torch.Tensor,
    target_shape: tuple[int, ...],
) -> torch.Tensor:
    if tuple(source.shape) == target_shape:
        return source
    output = source.float()
    while output.ndim < len(target_shape):
        output = output.unsqueeze(0)
    while output.ndim > len(target_shape):
        if output.shape[0] != 1:
            raise ValueError(
                f"Cannot reduce tensor rank from {tuple(source.shape)} to {target_shape}."
            )
        output = output.squeeze(0)
    for dimension, new_size in enumerate(target_shape):
        if output.shape[dimension] == new_size:
            continue
        permutation = [
            index for index in range(output.ndim) if index != dimension
        ] + [dimension]
        inverse = [0] * output.ndim
        for index, source_dimension in enumerate(permutation):
            inverse[source_dimension] = index
        output = output.permute(*permutation).contiguous()
        prefix = output.shape[:-1]
        output = _interpolate_last_dim(output, new_size).reshape(*prefix, new_size)
        output = output.permute(*inverse).contiguous()
    if tuple(output.shape) != target_shape:
        raise RuntimeError(
            f"Interpolation produced {tuple(output.shape)}, expected {target_shape}."
        )
    return output.to(dtype=source.dtype)


def build_action_dit_backbone_payload(
    model_config: DreamWAMConfig,
    wan_root: Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[str, Any], ActionBackboneReport]:
    """Derive FastWAM's ActionDiT backbone from Wan2.2 VideoDiT weights."""

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested for ActionDiT preparation but is unavailable."
        )
    with torch.device("meta"):
        action_expert = ActionDiT(
            action_dim=model_config.action_dim,
            hidden_dim=model_config.action_hidden_dim,
            ffn_dim=model_config.action_ffn_dim,
            text_dim=model_config.text_dim,
            freq_dim=model_config.freq_dim,
            num_heads=model_config.num_heads,
            attn_head_dim=model_config.attn_head_dim,
            num_layers=model_config.num_layers,
            eps=model_config.eps,
        )
    action_state = action_expert.state_dict()
    backbone_keys = {
        key
        for key in action_state
        if not key.startswith(ACTION_BACKBONE_SKIP_PREFIXES)
    }
    weight_map = _load_wan_weight_map(wan_root)
    missing = sorted(backbone_keys - weight_map.keys())
    if missing:
        raise KeyError(f"Wan VideoDiT cannot initialize ActionDiT: {missing[:10]}")

    backbone: dict[str, torch.Tensor] = {}
    copied = 0
    interpolated = 0
    for key, source_cpu in _iter_wan_tensors(wan_root, weight_map, backbone_keys):
        # FastWAM casts Wan weights to the model dtype before interpolation.
        source = source_cpu.to(device=device, dtype=dtype)
        target_shape = tuple(action_state[key].shape)
        if tuple(source.shape) == target_shape:
            value = source
            copied += 1
        else:
            value = _resize_tensor_to_shape(source, target_shape)
            if source.ndim >= 2 and source.shape[-1] != target_shape[-1]:
                alpha = (float(source.shape[-1]) / float(target_shape[-1])) ** 0.5
                value = value.float() * alpha
            interpolated += 1
        backbone[key] = value.detach().to(device="cpu", dtype=dtype).contiguous()

    payload: dict[str, Any] = {
        "policy": {
            "skip_prefixes": list(FASTWAM_ACTION_PAYLOAD_SKIP_PREFIXES),
            "alpha_scaling": True,
            "interpolation": "sequential_1d_linear_align_corners_true",
        },
        "backbone_state_dict": backbone,
        "meta": {
            "hidden_dim": int(model_config.action_hidden_dim),
            "ffn_dim": int(model_config.action_ffn_dim),
            "num_layers": int(model_config.num_layers),
            "num_heads": int(model_config.num_heads),
            "attn_head_dim": int(model_config.attn_head_dim),
            "text_dim": int(model_config.text_dim),
            "freq_dim": int(model_config.freq_dim),
            "eps": float(model_config.eps),
        },
    }
    return payload, ActionBackboneReport(
        copied_tensors=copied,
        interpolated_tensors=interpolated,
    )
