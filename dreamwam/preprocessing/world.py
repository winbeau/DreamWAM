from collections.abc import Iterable
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def _unit_frame_batch(videos: torch.Tensor) -> torch.Tensor:
    if videos.ndim != 5 or videos.shape[1] != 3:
        raise ValueError(
            f"videos must be [B,3,T,H,W], got {tuple(videos.shape)}."
        )
    batch, _, frames, height, width = videos.shape
    return (
        ((videos.detach().float().clamp(-1.0, 1.0) + 1.0) * 0.5)
        .permute(0, 2, 1, 3, 4)
        .reshape(batch * frames, 3, height, width)
        .contiguous()
    )


def _resize_to_patch_multiple(
    frames: torch.Tensor,
    patch_size: int,
) -> torch.Tensor:
    if frames.ndim != 4:
        raise ValueError(f"frames must be [N,3,H,W], got {tuple(frames.shape)}.")
    height = (frames.shape[-2] // patch_size) * patch_size
    width = (frames.shape[-1] // patch_size) * patch_size
    if height <= 0 or width <= 0:
        raise ValueError(
            f"frame size {tuple(frames.shape[-2:])} is smaller than patch {patch_size}."
        )
    if (height, width) == tuple(frames.shape[-2:]):
        return frames
    return F.interpolate(
        frames.float(),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )


class DINOExtractor(nn.Module):
    def __init__(self, model: nn.Module, device: torch.device):
        super().__init__()
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.patch_size = 14
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    @torch.no_grad()
    def forward(self, videos: torch.Tensor) -> torch.Tensor:
        if videos.shape[2] != 9:
            raise ValueError(
                f"DreamWAM DINO preprocessing requires 9 frames, got {videos.shape[2]}."
            )
        batch, frames = videos.shape[0], videos.shape[2]
        images = _resize_to_patch_multiple(
            _unit_frame_batch(videos).to(self.device),
            self.patch_size,
        )
        images = (
            images - self.mean.to(device=self.device, dtype=images.dtype)
        ) / self.std.to(device=self.device, dtype=images.dtype)
        output = self.model.forward_features(images)
        if not isinstance(output, dict) or "x_norm_patchtokens" not in output:
            raise KeyError("DINO forward_features must return x_norm_patchtokens.")
        tokens = output["x_norm_patchtokens"]
        height = images.shape[-2] // self.patch_size
        width = images.shape[-1] // self.patch_size
        expected_tokens = height * width
        if tokens.ndim != 3 or tokens.shape[1] != expected_tokens:
            raise ValueError(
                f"DINO patch token shape mismatch: {tuple(tokens.shape)}, "
                f"expected sequence {expected_tokens}."
            )
        return (
            tokens.reshape(batch, frames, height, width, tokens.shape[-1])
            .detach()
            .cpu()
            .contiguous()
        )


class DA3DepthExtractor(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        forward_group_batch_size: int = 64,
        video_batch_size: int = 8,
    ):
        super().__init__()
        if forward_group_batch_size <= 0 or video_batch_size <= 0:
            raise ValueError("Depth preprocessing batch sizes must be positive.")
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.patch_size = 14
        self.forward_group_batch_size = int(forward_group_batch_size)
        self.video_batch_size = int(video_batch_size)
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1),
            persistent=False,
        )

    def _resize_groups(self, groups: torch.Tensor) -> torch.Tensor:
        groups_count, views, channels, height, width = groups.shape
        resized = _resize_to_patch_multiple(
            groups.reshape(groups_count * views, channels, height, width),
            self.patch_size,
        )
        return resized.reshape(
            groups_count,
            views,
            channels,
            resized.shape[-2],
            resized.shape[-1],
        ).contiguous()

    @staticmethod
    def _depth_from_output(output) -> torch.Tensor:
        depth = output.get("depth") if hasattr(output, "get") else None
        if depth is None:
            raise KeyError("DA3 output is missing depth.")
        if depth.ndim == 5 and depth.shape[-1] == 1:
            depth = depth.squeeze(-1)
        if depth.ndim != 4:
            raise ValueError(
                f"DA3 depth must be [G,N,H,W], got {tuple(depth.shape)}."
            )
        return depth.float()

    @torch.no_grad()
    def _forward_groups(self, groups: torch.Tensor) -> torch.Tensor:
        chunks = []
        for start in range(0, groups.shape[0], self.forward_group_batch_size):
            chunk = groups[start : start + self.forward_group_batch_size].to(
                self.device,
                non_blocking=True,
            )
            chunk = (
                chunk - self.mean.to(device=self.device, dtype=chunk.dtype)
            ) / self.std.to(device=self.device, dtype=chunk.dtype)
            if self.device.type == "cuda":
                autocast_dtype = (
                    torch.bfloat16
                    if torch.cuda.is_bf16_supported()
                    else torch.float16
                )
                with torch.autocast("cuda", dtype=autocast_dtype):
                    output = self.model(
                        chunk,
                        None,
                        None,
                        export_feat_layers=[],
                        infer_gs=False,
                    )
            else:
                output = self.model(
                    chunk,
                    None,
                    None,
                    export_feat_layers=[],
                    infer_gs=False,
                )
            chunks.append(self._depth_from_output(output).detach().cpu())
        return torch.cat(chunks, dim=0)

    @torch.no_grad()
    def forward(self, videos: torch.Tensor) -> torch.Tensor:
        if videos.ndim != 5 or videos.shape[1] != 3:
            raise ValueError(
                f"videos must be [B,3,T,H,W], got {tuple(videos.shape)}."
            )
        if videos.shape[2] != 9:
            raise ValueError(
                f"DreamWAM depth preprocessing requires 9 frames, got {videos.shape[2]}."
            )
        if videos.shape[-1] % 2:
            raise ValueError("Two-camera horizontal video width must be even.")

        batches = [
            self._forward_video_batch(
                videos[start : start + self.video_batch_size]
            )
            for start in range(0, videos.shape[0], self.video_batch_size)
        ]
        return torch.cat(batches, dim=0)

    def _forward_video_batch(self, videos: torch.Tensor) -> torch.Tensor:
        batch, _, frames, height, width = videos.shape
        images = (
            ((videos.detach().float().clamp(-1.0, 1.0) + 1.0) * 0.5)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
        )
        middle = width // 2
        groups = torch.stack(
            [images[..., :middle], images[..., middle:]],
            dim=2,
        ).reshape(batch * frames, 2, 3, height, middle)
        groups = self._resize_groups(groups)
        depth = self._forward_groups(groups)
        if depth.shape[:2] != (batch * frames, 2):
            raise ValueError(
                f"DA3 returned wrong group/view shape: {tuple(depth.shape)}."
            )
        depth = depth.reshape(batch, frames, 2, depth.shape[-2], depth.shape[-1])
        depth = torch.cat([depth[:, :, 0], depth[:, :, 1]], dim=-1)
        return depth.unsqueeze(-1).contiguous()


def fit_rank8_pca(
    features: Iterable[torch.Tensor],
    *,
    max_tokens: int = 200000,
) -> dict[str, torch.Tensor]:
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive.")
    flattened = [
        feature.detach().float().cpu().reshape(-1, feature.shape[-1])
        for feature in features
    ]
    if not flattened:
        raise ValueError("PCA fitting requires at least one feature tensor.")
    values = torch.cat(flattened, dim=0)
    if values.shape[0] > max_tokens:
        generator = torch.Generator().manual_seed(0)
        indices = torch.randperm(values.shape[0], generator=generator)[:max_tokens]
        values = values[indices]
    if min(values.shape) < 8:
        raise ValueError(
            f"rank-8 PCA requires at least 8 samples and channels, got {tuple(values.shape)}."
        )
    mean = values.mean(dim=0)
    _, _, vectors = torch.pca_lowrank(values - mean, q=8, center=False)
    return {
        "mean": mean.contiguous(),
        "components": vectors[:, :8].T.contiguous(),
    }


def _project_rank8(
    raw: torch.Tensor,
    projection: dict[str, torch.Tensor],
) -> torch.Tensor:
    if raw.ndim != 4:
        raise ValueError(f"raw features must be [T,H,W,C], got {tuple(raw.shape)}.")
    if set(projection) != {"mean", "components"}:
        raise ValueError("projection must contain exactly mean and components.")
    values = raw.float().reshape(-1, raw.shape[-1])
    mean = projection["mean"].to(values)
    components = projection["components"].to(values)
    if components.shape != (8, values.shape[1]) or mean.shape != (values.shape[1],):
        raise ValueError(
            f"invalid rank-8 projection shapes: {tuple(mean.shape)}, "
            f"{tuple(components.shape)} for input {values.shape[1]}."
        )
    projected = (values - mean) @ components.T
    return projected.reshape(*raw.shape[:-1], 8)


def load_rank8_projection(
    path: str | Path,
    *,
    input_dim: int,
) -> dict[str, torch.Tensor]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing rank-8 projection: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError(f"Projection must be a mapping: {path}")
    required = {"mean", "components"}
    if not required.issubset(payload):
        raise KeyError(
            f"Projection {path} is missing keys: {sorted(required - payload.keys())}"
        )
    rank = payload.get("rank")
    if rank is not None and int(rank) != 8:
        raise ValueError(f"Projection {path} has rank {int(rank)}, expected 8.")
    mean = payload["mean"]
    components = payload["components"]
    if not isinstance(mean, torch.Tensor) or not isinstance(
        components, torch.Tensor
    ):
        raise TypeError(f"Projection tensors are invalid: {path}")
    if mean.shape != (input_dim,) or components.shape != (8, input_dim):
        raise ValueError(
            f"Projection {path} has shapes {tuple(mean.shape)} and "
            f"{tuple(components.shape)}, expected ({input_dim},) and (8,{input_dim})."
        )
    if not torch.isfinite(mean).all() or not torch.isfinite(components).all():
        raise FloatingPointError(f"Projection contains non-finite values: {path}")
    return {
        "mean": mean.float().contiguous(),
        "components": components.float().contiguous(),
    }


def _align_9_to_3(features: torch.Tensor) -> torch.Tensor:
    if features.ndim != 4 or features.shape[0] != 9:
        raise ValueError(
            f"temporal alignment requires [9,H,W,C], got {tuple(features.shape)}."
        )
    return torch.cat(
        [
            features[:1],
            features[1:5].mean(dim=0, keepdim=True),
            features[5:9].mean(dim=0, keepdim=True),
        ],
        dim=0,
    )


def _normalize_channel_first(features: torch.Tensor) -> torch.Tensor:
    mean = features.mean(dim=(1, 2, 3), keepdim=True)
    std = features.std(
        dim=(1, 2, 3),
        keepdim=True,
        unbiased=False,
    ).clamp_min(1.0e-6)
    return (features - mean) / std


def _patchify_depth(
    raw: torch.Tensor,
    *,
    latent_height: int,
    latent_width: int,
) -> torch.Tensor:
    if raw.ndim != 4 or raw.shape[0] != 9 or raw.shape[-1] != 1:
        raise ValueError(
            f"raw depth must be [9,H,W,1], got {tuple(raw.shape)}."
        )
    height, width = raw.shape[1:3]
    if height % latent_height or width % latent_width:
        raise ValueError(
            f"depth grid {(height, width)} must divide latent grid "
            f"{(latent_height, latent_width)}."
        )
    patch_height = height // latent_height
    patch_width = width // latent_width
    log_depth = torch.log(raw[..., 0].float().clamp_min(1.0e-6))
    return (
        log_depth.reshape(
            9,
            latent_height,
            patch_height,
            latent_width,
            patch_width,
        )
        .permute(0, 1, 3, 2, 4)
        .reshape(9, latent_height, latent_width, patch_height * patch_width)
        .contiguous()
    )


def fit_dino_projection(
    raw_batches: Iterable[torch.Tensor],
    *,
    max_tokens: int = 200000,
) -> dict[str, torch.Tensor]:
    samples = (
        sample
        for batch in raw_batches
        for sample in batch
    )
    return fit_rank8_pca(samples, max_tokens=max_tokens)


def fit_depth_projection(
    raw_batches: Iterable[torch.Tensor],
    *,
    latent_height: int,
    latent_width: int,
    max_tokens: int = 200000,
) -> dict[str, torch.Tensor]:
    samples = (
        _patchify_depth(
            sample,
            latent_height=latent_height,
            latent_width=latent_width,
        )
        for batch in raw_batches
        for sample in batch
    )
    return fit_rank8_pca(samples, max_tokens=max_tokens)


def dino_latents(
    raw_batch: torch.Tensor,
    projection: dict[str, torch.Tensor],
    *,
    latent_height: int,
    latent_width: int,
) -> torch.Tensor:
    latents = []
    for raw in raw_batch:
        projected = _project_rank8(raw, projection)
        resized = F.interpolate(
            projected.permute(0, 3, 1, 2),
            size=(latent_height, latent_width),
            mode="bilinear",
            align_corners=False,
        ).permute(0, 2, 3, 1)
        aligned = _align_9_to_3(resized)
        latents.append(
            _normalize_channel_first(
                aligned.permute(3, 0, 1, 2).contiguous()
            ).half()
        )
    return torch.stack(latents, dim=0).cpu().contiguous()


def depth_latents(
    raw_batch: torch.Tensor,
    projection: dict[str, torch.Tensor],
    *,
    latent_height: int,
    latent_width: int,
) -> torch.Tensor:
    latents = []
    for raw in raw_batch:
        patch_tokens = _patchify_depth(
            raw,
            latent_height=latent_height,
            latent_width=latent_width,
        )
        projected = _project_rank8(patch_tokens, projection)
        aligned = _align_9_to_3(projected)
        latents.append(
            _normalize_channel_first(
                aligned.permute(3, 0, 1, 2).contiguous()
            ).half()
        )
    return torch.stack(latents, dim=0).cpu().contiguous()
