from collections.abc import Iterable

import torch
from torch import nn

from ..normalization import LiberoNormalizer
from .flow import FlowLatentEncoder
from .world import (
    DA3DepthExtractor,
    DINOExtractor,
    depth_latents,
    dino_latents,
    fit_depth_projection,
    fit_dino_projection,
)


class DreamWAMPreprocessor(nn.Module):
    def __init__(
        self,
        *,
        flow_encoder: FlowLatentEncoder,
        dino_extractor: DINOExtractor,
        depth_extractor: DA3DepthExtractor,
        normalizer: LiberoNormalizer,
    ):
        super().__init__()
        self.flow_encoder = flow_encoder
        self.dino_extractor = dino_extractor
        self.depth_extractor = depth_extractor
        self.normalizer = normalizer

    @staticmethod
    def _validate_videos(videos: torch.Tensor) -> None:
        if videos.ndim != 5 or videos.shape[1] != 3:
            raise ValueError(
                f"videos must be [B,3,9,H,W], got {tuple(videos.shape)}."
            )
        if videos.shape[2] != 9:
            raise ValueError(
                f"DreamWAM preprocessing requires 9 frames, got {videos.shape[2]}."
            )
        if videos.shape[0] <= 0:
            raise ValueError("videos batch cannot be empty.")

    @torch.no_grad()
    def fit_projections(
        self,
        calibration_batches: Iterable[torch.Tensor],
        *,
        max_tokens: int = 200000,
    ) -> dict[str, dict[str, torch.Tensor]]:
        dino_raw_batches = []
        depth_raw_batches = []
        latent_size = None

        for videos in calibration_batches:
            self._validate_videos(videos)
            video_latents = self.flow_encoder.encode_video(videos)
            current_size = tuple(int(value) for value in video_latents.shape[-2:])
            if latent_size is None:
                latent_size = current_size
            elif latent_size != current_size:
                raise ValueError(
                    f"calibration latent size changed: {latent_size} vs {current_size}."
                )
            dino_raw_batches.append(self.dino_extractor(videos))
            depth_raw_batches.append(self.depth_extractor(videos))

        if latent_size is None:
            raise ValueError("calibration_batches is empty.")
        latent_height, latent_width = latent_size
        return {
            "dino": fit_dino_projection(
                dino_raw_batches,
                max_tokens=max_tokens,
            ),
            "depth": fit_depth_projection(
                depth_raw_batches,
                latent_height=latent_height,
                latent_width=latent_width,
                max_tokens=max_tokens,
            ),
        }

    @torch.no_grad()
    def preprocess_batch(
        self,
        *,
        videos: torch.Tensor,
        action: torch.Tensor,
        proprio: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        projections: dict[str, dict[str, torch.Tensor]],
        action_is_pad: torch.Tensor,
        image_is_pad: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        self._validate_videos(videos)
        if set(projections) != {"dino", "depth"}:
            raise ValueError("projections must contain exactly dino and depth.")
        batch_size = videos.shape[0]
        if action.ndim != 3 or action.shape[0] != batch_size:
            raise ValueError(
                f"action must be [B,T,D], got {tuple(action.shape)}."
            )
        if context.ndim != 3 or context.shape[0] != batch_size:
            raise ValueError(
                f"context must be [B,L,D], got {tuple(context.shape)}."
            )
        if context_mask.shape != context.shape[:2]:
            raise ValueError(
                f"context_mask must be [B,L], got {tuple(context_mask.shape)}."
            )
        if action_is_pad.shape != action.shape[:2]:
            raise ValueError(
                f"action_is_pad must be [B,T], got {tuple(action_is_pad.shape)}."
            )
        if proprio.ndim != 3 or proprio.shape != (
            batch_size,
            action.shape[1],
            8,
        ):
            raise ValueError(
                f"proprio must be [B,T,8], got {tuple(proprio.shape)}."
            )
        if image_is_pad.shape != (batch_size, videos.shape[2]):
            raise ValueError(
                f"image_is_pad must be [B,9], got {tuple(image_is_pad.shape)}."
            )

        video_latents = self.flow_encoder.encode_video(videos)
        flow_latents = self.flow_encoder(videos)
        if video_latents.shape != flow_latents.shape:
            raise ValueError(
                "Video and Flow VAE latents must have identical shapes, "
                f"got {tuple(video_latents.shape)} and {tuple(flow_latents.shape)}."
            )
        if video_latents.shape[2] != 3:
            raise ValueError(
                f"Wan VAE must map 9 frames to 3 latent steps, got {video_latents.shape[2]}."
            )
        latent_height, latent_width = video_latents.shape[-2:]

        dino = dino_latents(
            self.dino_extractor(videos),
            projections["dino"],
            latent_height=latent_height,
            latent_width=latent_width,
        )
        depth = depth_latents(
            self.depth_extractor(videos),
            projections["depth"],
            latent_height=latent_height,
            latent_width=latent_width,
        )
        expected_world_shape = (
            batch_size,
            8,
            video_latents.shape[2],
            latent_height,
            latent_width,
        )
        if tuple(dino.shape) != expected_world_shape:
            raise ValueError(
                f"DINO latent shape mismatch: {tuple(dino.shape)} vs {expected_world_shape}."
            )
        if tuple(depth.shape) != expected_world_shape:
            raise ValueError(
                f"Depth latent shape mismatch: {tuple(depth.shape)} vs {expected_world_shape}."
            )

        normalized_action = self.normalizer.normalize_action(action.float())
        normalized_proprio = self.normalizer.normalize_state(proprio.float())
        batch = {
            "video_latents": video_latents,
            "flow_latents": flow_latents,
            "dino_latents": dino,
            "depth_latents": depth,
            "action": normalized_action.detach().cpu().contiguous(),
            "action_is_pad": action_is_pad.detach().bool().cpu().contiguous(),
            "proprio": normalized_proprio.detach().cpu().contiguous(),
            "image_is_pad": image_is_pad.detach().bool().cpu().contiguous(),
            "context": context.detach().cpu().contiguous(),
            "context_mask": context_mask.detach().bool().cpu().contiguous(),
        }
        return batch
