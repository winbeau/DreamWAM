from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class InputPadder:
    def __init__(self, shape: torch.Size):
        height, width = shape[-2:]
        pad_height = (((height // 8) + 1) * 8 - height) % 8
        pad_width = (((width // 8) + 1) * 8 - width) % 8
        self.padding = [
            pad_width // 2,
            pad_width - pad_width // 2,
            pad_height // 2,
            pad_height - pad_height // 2,
        ]

    def pad(self, *inputs: torch.Tensor) -> list[torch.Tensor]:
        return [F.pad(value, self.padding, mode="replicate") for value in inputs]

    def unpad(self, value: torch.Tensor) -> torch.Tensor:
        height, width = value.shape[-2:]
        top = self.padding[2]
        bottom = height - self.padding[3]
        left = self.padding[0]
        right = width - self.padding[1]
        return value[..., top:bottom, left:right]


def normalized_videos_to_uint8(videos: torch.Tensor) -> torch.Tensor:
    if videos.ndim != 5 or videos.shape[1] != 3:
        raise ValueError(
            f"videos must be [B,3,T,H,W], got {tuple(videos.shape)}."
        )
    rgb = ((videos.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 127.5).round()
    return rgb.to(torch.uint8).permute(0, 2, 3, 4, 1).contiguous()


def _flow_colorwheel() -> np.ndarray:
    segments = (15, 6, 4, 11, 13, 6)
    colorwheel = np.zeros((sum(segments), 3))
    column = 0
    red_yellow, yellow_green, green_cyan, cyan_blue, blue_magenta, magenta_red = segments

    colorwheel[column : column + red_yellow, 0] = 255
    colorwheel[column : column + red_yellow, 1] = np.floor(
        255 * np.arange(red_yellow) / red_yellow
    )
    column += red_yellow
    colorwheel[column : column + yellow_green, 0] = 255 - np.floor(
        255 * np.arange(yellow_green) / yellow_green
    )
    colorwheel[column : column + yellow_green, 1] = 255
    column += yellow_green
    colorwheel[column : column + green_cyan, 1] = 255
    colorwheel[column : column + green_cyan, 2] = np.floor(
        255 * np.arange(green_cyan) / green_cyan
    )
    column += green_cyan
    colorwheel[column : column + cyan_blue, 1] = 255 - np.floor(
        255 * np.arange(cyan_blue) / cyan_blue
    )
    colorwheel[column : column + cyan_blue, 2] = 255
    column += cyan_blue
    colorwheel[column : column + blue_magenta, 2] = 255
    colorwheel[column : column + blue_magenta, 0] = np.floor(
        255 * np.arange(blue_magenta) / blue_magenta
    )
    column += blue_magenta
    colorwheel[column : column + magenta_red, 2] = 255 - np.floor(
        255 * np.arange(magenta_red) / magenta_red
    )
    colorwheel[column : column + magenta_red, 0] = 255
    return colorwheel


def flow_to_rgb(flow: torch.Tensor) -> torch.Tensor:
    if flow.ndim != 4 or flow.shape[-1] != 2:
        raise ValueError(f"flow must be [N,H,W,2], got {tuple(flow.shape)}.")
    colorwheel = _flow_colorwheel()
    columns = colorwheel.shape[0]
    images = []
    for field in flow.detach().float().cpu().numpy():
        horizontal = field[..., 0]
        vertical = field[..., 1]
        radius = np.sqrt(np.square(horizontal) + np.square(vertical))
        maximum = np.max(radius)
        horizontal = horizontal / (maximum + 1.0e-5)
        vertical = vertical / (maximum + 1.0e-5)
        radius = np.sqrt(np.square(horizontal) + np.square(vertical))
        angle = np.arctan2(-vertical, -horizontal) / np.pi
        position = (angle + 1.0) * 0.5 * (columns - 1)
        lower = np.floor(position).astype(np.int32)
        upper = lower + 1
        upper[upper == columns] = 0
        fraction = position - lower
        image = np.zeros((*horizontal.shape, 3), dtype=np.uint8)
        for channel_index in range(3):
            values = colorwheel[:, channel_index]
            color = (1.0 - fraction) * (values[lower] / 255.0)
            color += fraction * (values[upper] / 255.0)
            in_range = radius <= 1
            color[in_range] = 1.0 - radius[in_range] * (
                1.0 - color[in_range]
            )
            color[~in_range] *= 0.75
            image[..., channel_index] = np.floor(255.0 * color)
        images.append(torch.from_numpy(image))
    return torch.stack(images).contiguous()


def _vae_encode(
    vae: Any,
    videos: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    videos = videos.to(device=device, dtype=dtype)
    if hasattr(vae, "encode"):
        latents = vae.encode(videos, device=str(device))
    elif hasattr(vae, "single_encode"):
        latents = vae.single_encode(videos, device=str(device))
    else:
        raise TypeError("Wan VAE must provide single_encode or encode.")
    if isinstance(latents, list):
        latents = torch.stack(latents, dim=0)
    if not isinstance(latents, torch.Tensor) or latents.ndim != 5:
        raise TypeError(
            "Wan VAE must return [B,C,T,H,W] tensor or a list of [C,T,H,W] tensors."
        )
    if latents.shape[0] != videos.shape[0]:
        raise ValueError(
            f"VAE batch mismatch: {latents.shape[0]} vs {videos.shape[0]}."
        )
    return latents.detach().cpu().contiguous()


class FlowLatentEncoder(nn.Module):
    def __init__(
        self,
        *,
        raft_model: nn.Module,
        vae: Any,
        device: torch.device,
        vae_dtype: torch.dtype,
        raft_iterations: int = 20,
        raft_video_batch_size: int = 4,
        flow_vae_batch_size: int = 1,
    ):
        super().__init__()
        if raft_iterations <= 0:
            raise ValueError("raft_iterations must be positive.")
        if raft_video_batch_size <= 0 or flow_vae_batch_size <= 0:
            raise ValueError("Flow preprocessing batch sizes must be positive.")
        self.raft_model = raft_model.to(device).eval()
        self.vae = vae
        self.device = torch.device(device)
        self.vae_dtype = vae_dtype
        self.raft_iterations = int(raft_iterations)
        self.raft_video_batch_size = int(raft_video_batch_size)
        self.flow_vae_batch_size = int(flow_vae_batch_size)

    @torch.no_grad()
    def encode_video(self, videos: torch.Tensor) -> torch.Tensor:
        if videos.ndim != 5 or videos.shape[1] != 3:
            raise ValueError(
                f"videos must be [B,3,T,H,W], got {tuple(videos.shape)}."
            )
        return _vae_encode(
            self.vae,
            videos,
            device=self.device,
            dtype=self.vae_dtype,
        )

    @torch.no_grad()
    def forward(self, videos: torch.Tensor) -> torch.Tensor:
        if videos.ndim != 5 or videos.shape[1] != 3:
            raise ValueError(
                f"videos must be [B,3,T,H,W], got {tuple(videos.shape)}."
            )
        if videos.shape[2] != 9:
            raise ValueError(
                f"DreamWAM preprocessing requires 9 frames, got {videos.shape[2]}."
            )

        latent_batches = []
        for start in range(0, videos.shape[0], self.raft_video_batch_size):
            flow_videos = self._flow_videos(
                videos[start : start + self.raft_video_batch_size]
            )
            for vae_start in range(0, flow_videos.shape[0], self.flow_vae_batch_size):
                latent_batches.append(
                    _vae_encode(
                        self.vae,
                        flow_videos[
                            vae_start : vae_start + self.flow_vae_batch_size
                        ],
                        device=self.device,
                        dtype=self.vae_dtype,
                    )
                )
        return torch.cat(latent_batches, dim=0)

    def _flow_videos(self, videos: torch.Tensor) -> torch.Tensor:
        rgb = normalized_videos_to_uint8(videos)
        batch, frames, height, width, _ = rgb.shape
        image1 = rgb[:, :-1].reshape(-1, height, width, 3)
        image2 = rgb[:, 1:].reshape(-1, height, width, 3)
        image1 = image1.permute(0, 3, 1, 2).float().to(self.device)
        image2 = image2.permute(0, 3, 1, 2).float().to(self.device)
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        _, flow = self.raft_model(
            image1,
            image2,
            iters=self.raft_iterations,
            test_mode=True,
        )
        if not isinstance(flow, torch.Tensor) or flow.ndim != 4 or flow.shape[1] != 2:
            raise ValueError(
                f"RAFT must return flow [N,2,H,W], got {type(flow)} "
                f"{getattr(flow, 'shape', None)}."
            )
        flow = padder.unpad(flow).detach().cpu().permute(0, 2, 3, 1)
        flow_images = flow_to_rgb(flow).reshape(
            batch,
            frames - 1,
            height,
            width,
            3,
        )
        flow_images = torch.cat(
            [flow_images[:, :1].clone(), flow_images],
            dim=1,
        )
        flow_videos = (
            flow_images.float()
            .mul(2.0 / 255.0)
            .sub(1.0)
            .permute(0, 4, 1, 2, 3)
            .contiguous()
        )
        return flow_videos
