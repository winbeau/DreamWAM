import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file as load_safetensors

from .config import ReleaseConfig
from .normalization import LiberoNormalizer
from .preprocessing import (
    DA3DepthExtractor,
    DINOExtractor,
    DreamWAMPreprocessor,
    FlowLatentEncoder,
)
from .wan import HuggingfaceTokenizer, WanTextEncoder, WanVideoVAE38


def _resolve_file(root: Path, names: tuple[str, ...]) -> Path:
    if root.is_file():
        if root.name not in names:
            raise ValueError(f"Unexpected checkpoint filename: {root}")
        return root
    if not root.is_dir():
        raise FileNotFoundError(f"Missing pretrained component directory: {root}")
    matches = [root / name for name in names if (root / name).is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one of {names} under {root}, found {matches}."
        )
    return matches[0]


def _load_state(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        state = load_safetensors(str(path), device="cpu")
    else:
        state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
        state = state["state_dict"]
    if isinstance(state, dict) and isinstance(state.get("model_state"), dict):
        state = state["model_state"]
    if not isinstance(state, dict) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in state.items()
    ):
        raise TypeError(f"Checkpoint is not a tensor state dictionary: {path}")
    return state


def _materialize_from_state(
    model: torch.nn.Module,
    state: dict[str, torch.Tensor],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.nn.Module:
    incompatible = model.load_state_dict(state, strict=True, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Strict checkpoint loading returned incompatible keys: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}."
        )
    return model.to(device=device, dtype=dtype).eval()


def load_wan_vae(
    wan_root: str | Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> WanVideoVAE38:
    """Strict-load the frozen Wan2.2 VAE without a random fallback."""

    checkpoint = _resolve_file(
        Path(wan_root),
        ("Wan2.2_VAE.safetensors", "Wan2.2_VAE.pth"),
    )
    state = _load_state(checkpoint)
    if any(key.startswith("model.") for key in state):
        raise ValueError(
            f"Wan VAE checkpoint is already prefixed unexpectedly: {checkpoint}"
        )
    converted = {f"model.{key}": value for key, value in state.items()}
    with torch.device("meta"):
        model = WanVideoVAE38()
    model = _materialize_from_state(
        model,
        converted,
        device=device,
        dtype=dtype,
    )
    return model.requires_grad_(False)


class WanContextEncoder:
    """Strict-loaded frozen Wan UMT5 encoder and its local tokenizer."""

    def __init__(
        self,
        text_checkpoint: str | Path,
        tokenizer_path: str | Path,
        *,
        device: torch.device,
        dtype: torch.dtype,
        context_length: int = 128,
    ):
        checkpoint = _resolve_file(
            Path(text_checkpoint),
            (
                "models_t5_umt5-xxl-enc-bf16.safetensors",
                "models_t5_umt5-xxl-enc-bf16.pth",
            ),
        )
        tokenizer_path = Path(tokenizer_path)
        if not tokenizer_path.is_dir():
            raise FileNotFoundError(f"Missing Wan tokenizer: {tokenizer_path}")
        state = _load_state(checkpoint)
        with torch.device("meta"):
            model = WanTextEncoder()
        self.model = _materialize_from_state(
            model,
            state,
            device=device,
            dtype=dtype,
        ).requires_grad_(False)
        self.tokenizer = HuggingfaceTokenizer(
            name=str(tokenizer_path),
            seq_len=context_length,
            clean="whitespace",
            local_files_only=True,
        )
        self.device = torch.device(device)
        self.dtype = dtype

    @torch.no_grad()
    def __call__(self, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(prompts, list) or not prompts or not all(
            isinstance(prompt, str) and prompt for prompt in prompts
        ):
            raise ValueError("prompts must be a non-empty list of strings.")
        ids, mask = self.tokenizer(
            prompts,
            return_mask=True,
            add_special_tokens=True,
        )
        ids = ids.to(self.device)
        mask = mask.to(device=self.device, dtype=torch.bool)
        context = self.model(ids, mask)
        sequence_lengths = mask.sum(dim=1).long()
        for index, sequence_length in enumerate(sequence_lengths):
            context[index, sequence_length:] = 0
        return context.to(dtype=self.dtype), torch.ones_like(mask)


class _RAFTArgs(dict):
    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error


def load_raft(
    source_root: str | Path,
    checkpoint: str | Path,
    *,
    device: torch.device,
) -> torch.nn.Module:
    source_root = Path(source_root)
    core = source_root / "core"
    checkpoint = Path(checkpoint)
    if not (core / "raft.py").is_file():
        raise FileNotFoundError(f"Missing RAFT source: {core / 'raft.py'}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing RAFT checkpoint: {checkpoint}")
    sys.path.insert(0, str(core))
    try:
        from raft import RAFT
    finally:
        sys.path.pop(0)
    args = _RAFTArgs(
        small=False,
        mixed_precision=False,
        alternate_corr=False,
    )
    wrapped = torch.nn.DataParallel(RAFT(args))
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    wrapped.load_state_dict(state, strict=True)
    return wrapped.module.to(device).eval()


def load_dino(
    source_root: str | Path,
    checkpoint: str | Path,
    *,
    device: torch.device,
) -> torch.nn.Module:
    source_root = Path(source_root)
    checkpoint = Path(checkpoint)
    if not source_root.is_dir():
        raise FileNotFoundError(f"Missing DINOv2 source: {source_root}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing DINOv2 checkpoint: {checkpoint}")
    model = torch.hub.load(
        str(source_root),
        "dinov2_vitb14_reg",
        source="local",
        pretrained=False,
        verbose=False,
    )
    model.load_state_dict(_load_state(checkpoint), strict=True)
    model.head = torch.nn.Identity()
    return model.to(device).eval()


_DA3_ALLOWED_MISSING = frozenset(
    {
        "head.scratch.output_conv2_aux.1.2.bias",
        "head.scratch.output_conv2_aux.1.2.weight",
        "head.scratch.output_conv2_aux.2.2.bias",
        "head.scratch.output_conv2_aux.2.2.weight",
        "head.scratch.output_conv2_aux.3.2.bias",
        "head.scratch.output_conv2_aux.3.2.weight",
    }
)


def _strip_prefix(
    state: dict[str, torch.Tensor], prefix: str
) -> dict[str, torch.Tensor]:
    return {
        key[len(prefix):] if key.startswith(prefix) else key: value
        for key, value in state.items()
    }


def load_da3(
    source_root: str | Path,
    checkpoint_root: str | Path,
    *,
    model_name: str,
    device: torch.device,
) -> torch.nn.Module:
    source = Path(source_root) / "src"
    checkpoint_root = Path(checkpoint_root)
    if not source.is_dir():
        raise FileNotFoundError(f"Missing Depth Anything 3 source: {source}")
    sys.path.insert(0, str(source))
    try:
        from depth_anything_3.cfg import create_object, load_config
        from depth_anything_3.registry import MODEL_REGISTRY
        from depth_anything_3.utils.model_loading import convert_general_state_dict
    finally:
        sys.path.pop(0)
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown DA3 model {model_name!r}; available={list(MODEL_REGISTRY)}"
        )
    model = create_object(load_config(MODEL_REGISTRY[model_name]))
    checkpoint = _resolve_file(
        checkpoint_root,
        ("model.safetensors", "pytorch_model.bin", "model.pt"),
    )
    state = _load_state(checkpoint)
    model_keys = set(model.state_dict())
    converted = convert_general_state_dict(state)
    candidates = (
        state,
        _strip_prefix(state, "model."),
        _strip_prefix(state, "module."),
        converted,
        _strip_prefix(converted, "model."),
        _strip_prefix(converted, "module."),
    )
    selected = max(
        candidates,
        key=lambda candidate: len(model_keys.intersection(candidate)),
    )
    if not model_keys.intersection(selected):
        raise ValueError(f"DA3 checkpoint does not match model keys: {checkpoint}")
    incompatible = model.load_state_dict(selected, strict=False)
    unexpected = sorted(incompatible.unexpected_keys)
    unsupported_missing = sorted(
        set(incompatible.missing_keys) - _DA3_ALLOWED_MISSING
    )
    if unexpected or unsupported_missing:
        raise ValueError(
            "DA3 checkpoint is incompatible: "
            f"missing={unsupported_missing}, unexpected={unexpected}."
        )
    return model.to(device).eval()


@dataclass(frozen=True)
class PrecomputeComponents:
    preprocessor: DreamWAMPreprocessor
    encode_text: WanContextEncoder


def build_precompute_components(
    config: ReleaseConfig,
    *,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> PrecomputeComponents:
    device = torch.device(device)
    preprocessing: dict[str, Any] = config.preprocessing
    vae = load_wan_vae(
        preprocessing["wan_vae_checkpoint"],
        device=device,
        dtype=dtype,
    )
    flow_encoder = FlowLatentEncoder(
        raft_model=load_raft(
            preprocessing["raft_source"],
            preprocessing["raft_checkpoint"],
            device=device,
        ),
        vae=vae,
        device=device,
        vae_dtype=dtype,
        raft_iterations=int(preprocessing.get("raft_iterations", 20)),
        raft_video_batch_size=int(
            preprocessing.get("raft_video_batch_size", 4)
        ),
        flow_vae_batch_size=int(
            preprocessing.get("flow_vae_batch_size", 1)
        ),
    )
    preprocessor = DreamWAMPreprocessor(
        flow_encoder=flow_encoder,
        dino_extractor=DINOExtractor(
            load_dino(
                preprocessing["dino_source"],
                preprocessing["dino_checkpoint"],
                device=device,
            ),
            device,
        ),
        depth_extractor=DA3DepthExtractor(
            load_da3(
                preprocessing["depth_source"],
                preprocessing["depth_checkpoint"],
                model_name=str(preprocessing.get("depth_model_name", "da3-base")),
                device=device,
            ),
            device,
            forward_group_batch_size=int(
                preprocessing.get("depth_forward_group_batch_size", 64)
            ),
            video_batch_size=int(preprocessing.get("depth_video_batch_size", 8)),
        ),
        normalizer=LiberoNormalizer(config.paths.dataset_stats),
    )
    return PrecomputeComponents(
        preprocessor=preprocessor,
        encode_text=WanContextEncoder(
            preprocessing["wan_text_checkpoint"],
            preprocessing["wan_tokenizer"],
            device=device,
            dtype=dtype,
            context_length=int(preprocessing.get("context_length", 128)),
        ),
    )
