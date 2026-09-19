import math
from dataclasses import dataclass, replace

import torch
from torch import nn
from torch.nn import functional as F

from .experts import ActionDiT, VideoDiT
from .layers import PatchHead
from .mot import JointMoT
from .scheduler import ContinuousFlowMatchScheduler
from .sparse import SparseConfig


@dataclass(frozen=True)
class DreamWAMConfig:
    setting: str = "joint"
    video_latent_dim: int = 48
    flow_latent_dim: int = 48
    dino_latent_dim: int = 8
    depth_latent_dim: int = 8
    action_dim: int = 7
    proprio_dim: int = 8
    text_dim: int = 4096
    freq_dim: int = 256
    video_hidden_dim: int = 3072
    video_ffn_dim: int = 14336
    action_hidden_dim: int = 1024
    action_ffn_dim: int = 4096
    num_heads: int = 24
    attn_head_dim: int = 128
    num_layers: int = 30
    use_gradient_checkpointing: bool = True
    patch_size: tuple[int, int, int] = (1, 2, 2)
    routing_fusion: str = "sequential"
    routing_order: tuple[str, ...] = ("dino", "depth")
    dino_injection_layers: tuple[int, ...] = tuple(range(26, 30))
    depth_injection_layers: tuple[int, ...] = tuple(range(22, 30))
    local_preview_enabled: bool = False
    local_preview_loss_weight: float = 0.0
    local_preview_delta_scale: float = 1.0
    local_preview_layer_weighting: str = "uniform"
    local_preview_min_layer_weight: float = 0.25
    video_loss_weight: float = 1.0
    action_loss_weight: float = 1.0
    flow_loss_weight: float = 0.5
    dino_loss_weight: float = 0.5
    dino_final_loss_weight: float = 0.025
    depth_loss_weight: float = 0.5
    depth_final_loss_weight: float = 0.05
    gate_l1_weight: float = 1.0e-4
    vae_temporal_downsample_factor: int = 4
    eps: float = 1.0e-6


class GatedWorldRouter(nn.Module):

    def __init__(
        self,
        *,
        hidden_dim: int,
        dino_latent_dim: int,
        depth_latent_dim: int,
        patch_size: tuple[int, int, int],
        fusion_mode: str,
        routing_order: tuple[str, ...],
        injection_layers: dict[str, tuple[int, ...]],
        local_preview_enabled: bool,
        local_preview_delta_scale: float,
        eps: float,
    ):
        super().__init__()
        stream_names = ("dino", "depth")
        self.fusion_mode = str(fusion_mode).strip().lower()
        if self.fusion_mode not in {"sequential", "parallel"}:
            raise ValueError("fusion_mode must be 'sequential' or 'parallel'.")
        if tuple(sorted(routing_order)) != tuple(sorted(stream_names)):
            raise ValueError("routing_order must contain dino and depth exactly once.")
        self.stream_names = tuple(routing_order)
        self.local_preview_enabled = bool(local_preview_enabled)
        self.local_preview_delta_scale = float(local_preview_delta_scale)
        if self.local_preview_delta_scale <= 0:
            raise ValueError("local_preview_delta_scale must be positive.")
        if self.local_preview_enabled and self.fusion_mode != "parallel":
            raise ValueError("Local-preview supervision requires parallel fusion.")
        self.injection_layers = {
            name: frozenset(int(index) for index in injection_layers[name])
            for name in stream_names
        }
        if any(not layers for layers in self.injection_layers.values()):
            raise ValueError("Every residual route must contain injection layers.")
        stream_dims = {
            "dino": int(dino_latent_dim),
            "depth": int(depth_latent_dim),
        }
        if min(stream_dims.values()) <= 0:
            raise ValueError("DINO and depth latent dimensions must be positive.")

        self.heads = nn.ModuleDict(
            {
                name: PatchHead(
                    hidden_dim=hidden_dim,
                    out_dim=latent_dim,
                    patch_size=patch_size,
                    eps=eps,
                )
                for name, latent_dim in stream_dims.items()
            }
        )
        self.gate_norms = nn.ModuleDict(
            {name: nn.LayerNorm(hidden_dim) for name in self.stream_names}
        )
        self.gates = nn.ModuleDict(
            {name: nn.Linear(hidden_dim, 1) for name in self.stream_names}
        )
        bottleneck_dim = max(1, hidden_dim // 4)
        self.residual_branches = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, bottleneck_dim),
                    nn.GELU(),
                    nn.Linear(bottleneck_dim, hidden_dim),
                )
                for name in self.stream_names
            }
        )

        for name in self.stream_names:
            nn.init.zeros_(self.gates[name].weight)
            nn.init.constant_(self.gates[name].bias, -4.0)
            nn.init.zeros_(self.residual_branches[name][-1].weight)
            nn.init.zeros_(self.residual_branches[name][-1].bias)
        self._runtime_gates: dict[str, list[torch.Tensor]] = {
            name: [] for name in self.stream_names
        }
        self._runtime_local_preview_tokens: dict[
            str, list[tuple[int, torch.Tensor]]
        ] = {name: [] for name in self.stream_names}

    def reset_runtime(self) -> None:
        for gates in self._runtime_gates.values():
            gates.clear()
        for records in self._runtime_local_preview_tokens.values():
            records.clear()

    def gate_l1_loss(self) -> torch.Tensor:
        gate_means = [
            gate.mean()
            for gates in self._runtime_gates.values()
            for gate in gates
        ]
        if not gate_means:
            raise RuntimeError("Gate loss requested before residual gates were evaluated.")
        return torch.stack(gate_means).mean()

    @staticmethod
    def _context_summary(
        tokens: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        if context.ndim != 3 or context.shape[0] != tokens.shape[0]:
            raise ValueError(
                f"context must be [B,L,D], got {tuple(context.shape)}."
            )
        if context.shape[2] != tokens.shape[2]:
            raise ValueError(
                "Context and video token widths must match after projection: "
                f"{context.shape[2]} vs {tokens.shape[2]}."
            )
        if context_mask.ndim == 3:
            context_mask = context_mask.any(dim=1)
        if context_mask.shape != context.shape[:2]:
            raise ValueError(
                "context_mask must resolve to [B,L], "
                f"got {tuple(context_mask.shape)}."
            )
        valid = context_mask.to(
            device=context.device,
            dtype=context.dtype,
        ).unsqueeze(-1)
        denominator = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (context * valid).sum(dim=1, keepdim=True) / denominator

    def forward(
        self,
        layer_index: int,
        tokens: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        if not any(
            layer_index in layers for layers in self.injection_layers.values()
        ):
            return tokens

        context_summary = self._context_summary(tokens, context, context_mask)
        if self.fusion_mode == "parallel":
            base = tokens
            combined_delta = torch.zeros_like(base)
            for name in self.stream_names:
                if layer_index not in self.injection_layers[name]:
                    continue
                gate = torch.sigmoid(
                    self.gates[name](
                        self.gate_norms[name](base + context_summary)
                    )
                )
                self._runtime_gates[name].append(gate)
                residual = self.residual_branches[name](base)
                stream_delta = gate.to(dtype=base.dtype) * residual
                if self.local_preview_enabled:
                    stream_delta = self.local_preview_delta_scale * stream_delta
                    self._runtime_local_preview_tokens[name].append(
                        (layer_index, base + stream_delta)
                    )
                combined_delta = combined_delta + stream_delta
            return base + combined_delta

        routed = tokens
        for name in self.stream_names:
            if layer_index not in self.injection_layers[name]:
                continue
            gate = torch.sigmoid(
                self.gates[name](
                    self.gate_norms[name](routed + context_summary)
                )
            )
            self._runtime_gates[name].append(gate)
            residual = self.residual_branches[name](routed)
            routed = routed + gate.to(dtype=routed.dtype) * residual
        return routed

    def local_preview_records(
        self,
        name: str,
    ) -> tuple[tuple[int, torch.Tensor], ...]:
        if name not in self._runtime_local_preview_tokens:
            raise KeyError(f"Unknown world stream: {name}")
        return tuple(self._runtime_local_preview_tokens[name])

    def local_preview_layer_weights(
        self,
        count: int,
        *,
        mode: str,
        min_weight: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if count <= 0:
            raise ValueError("Local-preview loss requires at least one layer.")
        mode = str(mode).strip().lower()
        if mode == "uniform":
            return torch.ones(count, device=device, dtype=dtype)
        if mode == "cosine_ramp":
            if not 0.0 <= min_weight <= 1.0:
                raise ValueError("local-preview min_weight must be in [0,1].")
            if count == 1:
                return torch.ones(1, device=device, dtype=dtype)
            positions = torch.linspace(0.0, 1.0, count, device=device, dtype=dtype)
            ramp = 0.5 - 0.5 * torch.cos(math.pi * positions)
            return min_weight + (1.0 - min_weight) * ramp
        raise ValueError(f"Unsupported local-preview weighting: {mode}")

    def latest_gate_map(
        self,
        name: str,
        target_shape: tuple[int, ...],
    ) -> torch.Tensor:
        gates = self._runtime_gates.get(name)
        if not gates:
            raise RuntimeError(f"No gate activation was recorded for {name}.")
        if len(target_shape) != 5:
            raise ValueError(f"target_shape must be [B,C,T,H,W], got {target_shape}.")
        gate = gates[-1]
        batch, sequence, width = gate.shape
        if width != 1 or batch != target_shape[0]:
            raise ValueError(
                f"Gate shape {tuple(gate.shape)} does not match target {target_shape}."
            )
        patch_t, patch_h, patch_w = self.heads[name].patch_size
        target_t, target_h, target_w = target_shape[2:]
        if target_t % patch_t or target_h % patch_h or target_w % patch_w:
            raise ValueError(
                f"Target shape {target_shape} is not divisible by patch size "
                f"{self.heads[name].patch_size}."
            )
        grid = (
            target_t // patch_t,
            target_h // patch_h,
            target_w // patch_w,
        )
        if sequence != math.prod(grid):
            raise ValueError(
                f"Gate sequence has {sequence} tokens, expected {math.prod(grid)}."
            )
        gate_map = gate.transpose(1, 2).reshape(batch, 1, *grid)
        gate_map = gate_map.repeat_interleave(patch_t, dim=2)
        gate_map = gate_map.repeat_interleave(patch_h, dim=3)
        gate_map = gate_map.repeat_interleave(patch_w, dim=4)
        return gate_map[:, :, :target_t, :target_h, :target_w]

    def predict(
        self,
        *,
        tokens: torch.Tensor,
        time_embedding: torch.Tensor,
        video_expert: VideoDiT,
        grid_size: tuple[int, int, int],
    ) -> dict[str, torch.Tensor]:
        predictions = {}
        for name in self.stream_names:
            patches = self.heads[name](tokens, time_embedding)
            predictions[name] = video_expert.unpatchify(
                patches,
                grid_size,
                self.heads[name].proj.out_features
                // (
                    self.heads[name].patch_size[0]
                    * self.heads[name].patch_size[1]
                    * self.heads[name].patch_size[2]
                ),
            )
        return predictions

    def predict_local_previews(
        self,
        *,
        time_embedding: torch.Tensor,
        video_expert: VideoDiT,
        grid_size: tuple[int, int, int],
    ) -> dict[str, tuple[tuple[int, torch.Tensor], ...]]:
        predictions = {}
        for name in self.stream_names:
            stream_predictions = []
            for layer_index, tokens in self.local_preview_records(name):
                patches = self.heads[name](tokens, time_embedding)
                channels = self.heads[name].proj.out_features // math.prod(
                    self.heads[name].patch_size
                )
                stream_predictions.append(
                    (
                        layer_index,
                        video_expert.unpatchify(patches, grid_size, channels),
                    )
                )
            predictions[name] = tuple(stream_predictions)
        return predictions


class DreamWAMJoint(nn.Module):
    def __init__(self, config: DreamWAMConfig = DreamWAMConfig()):
        super().__init__()
        if config.setting not in {"uncond", "joint"}:
            raise ValueError("setting must be 'uncond' or 'joint'.")
        configured_layers = (
            *config.dino_injection_layers,
            *config.depth_injection_layers,
        )
        invalid_layers = [
            index
            for index in configured_layers
            if index < 0 or index >= config.num_layers
        ]
        if invalid_layers:
            raise ValueError(
                f"injection layers outside [0,{config.num_layers}): {invalid_layers}"
            )
        if config.gate_l1_weight < 0:
            raise ValueError("gate_l1_weight must be non-negative.")
        if config.local_preview_enabled:
            if config.routing_fusion != "parallel":
                raise ValueError("Local-preview supervision requires parallel fusion.")
            if config.local_preview_loss_weight <= 0:
                raise ValueError("local_preview_loss_weight must be positive.")
        self.config = config

        video_expert = VideoDiT(
            video_latent_dim=config.video_latent_dim,
            flow_latent_dim=config.flow_latent_dim,
            hidden_dim=config.video_hidden_dim,
            ffn_dim=config.video_ffn_dim,
            text_dim=config.text_dim,
            freq_dim=config.freq_dim,
            patch_size=config.patch_size,
            num_heads=config.num_heads,
            attn_head_dim=config.attn_head_dim,
            num_layers=config.num_layers,
            eps=config.eps,
        )
        action_expert = ActionDiT(
            action_dim=config.action_dim,
            hidden_dim=config.action_hidden_dim,
            ffn_dim=config.action_ffn_dim,
            text_dim=config.text_dim,
            freq_dim=config.freq_dim,
            num_heads=config.num_heads,
            attn_head_dim=config.attn_head_dim,
            num_layers=config.num_layers,
            eps=config.eps,
        )
        self.mot = JointMoT(
            video_expert,
            action_expert,
            action_attends_all_video=config.setting == "joint",
            use_gradient_checkpointing=config.use_gradient_checkpointing,
        )
        self.proprio_encoder = nn.Linear(config.proprio_dim, config.text_dim)
        self.world_residual = GatedWorldRouter(
            hidden_dim=config.video_hidden_dim,
            dino_latent_dim=config.dino_latent_dim,
            depth_latent_dim=config.depth_latent_dim,
            patch_size=config.patch_size,
            fusion_mode=config.routing_fusion,
            routing_order=config.routing_order,
            injection_layers={
                "dino": config.dino_injection_layers,
                "depth": config.depth_injection_layers,
            },
            local_preview_enabled=config.local_preview_enabled,
            local_preview_delta_scale=config.local_preview_delta_scale,
            eps=config.eps,
        )
        self.video_scheduler = ContinuousFlowMatchScheduler()
        self.action_scheduler = ContinuousFlowMatchScheduler()

    @property
    def video_expert(self) -> VideoDiT:
        return self.mot.video_expert

    @property
    def action_expert(self) -> ActionDiT:
        return self.mot.action_expert

    def _append_proprio_to_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if context.ndim != 3 or context_mask.shape != context.shape[:2]:
            raise ValueError(
                "context/context_mask must be [B,L,D]/[B,L], got "
                f"{tuple(context.shape)} and {tuple(context_mask.shape)}."
            )
        if context.shape[2] != self.config.text_dim:
            raise ValueError(
                f"context width must be {self.config.text_dim}, got {context.shape[2]}."
            )
        if proprio.ndim != 2 or proprio.shape != (
            context.shape[0],
            self.config.proprio_dim,
        ):
            raise ValueError(
                "proprio must be "
                f"[B,{self.config.proprio_dim}], got {tuple(proprio.shape)}."
            )
        proprio_token = self.proprio_encoder(
            proprio.to(device=context.device, dtype=context.dtype)
        ).unsqueeze(1)
        proprio_mask = torch.ones(
            context.shape[0],
            1,
            dtype=torch.bool,
            device=context_mask.device,
        )
        return (
            torch.cat([context, proprio_token], dim=1),
            torch.cat([context_mask.bool(), proprio_mask], dim=1),
        )

    def forward(
        self,
        *,
        video_latents: torch.Tensor,
        flow_latents: torch.Tensor,
        action_tokens: torch.Tensor,
        video_timestep: torch.Tensor,
        action_timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: torch.Tensor,
        apply_world_residual: bool = True,
        predict_world_targets: bool = True,
    ) -> dict:
        context, context_mask = self._append_proprio_to_context(
            context,
            context_mask,
            proprio,
        )
        return self._forward_conditioned(
            video_latents=video_latents,
            flow_latents=flow_latents,
            action_tokens=action_tokens,
            video_timestep=video_timestep,
            action_timestep=action_timestep,
            context=context,
            context_mask=context_mask,
            apply_world_residual=apply_world_residual,
            predict_world_targets=predict_world_targets,
        )

    def _forward_conditioned(
        self,
        *,
        video_latents: torch.Tensor,
        flow_latents: torch.Tensor,
        action_tokens: torch.Tensor,
        video_timestep: torch.Tensor,
        action_timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        apply_world_residual: bool = True,
        predict_world_targets: bool = True,
        sparse: "SparseConfig | None" = None,
        step_index: int = 0,
        num_steps: int = 1,
    ) -> dict:
        if predict_world_targets and not apply_world_residual:
            raise ValueError(
                "World-target prediction requires world residual injection."
            )
        self.world_residual.reset_runtime()
        video_state = self.video_expert.pre_dit(
            video_latents=video_latents,
            flow_latents=flow_latents,
            timestep=video_timestep,
            context=context,
            context_mask=context_mask,
        )
        action_state = self.action_expert.pre_dit(
            action_tokens=action_tokens,
            timestep=action_timestep,
            context=context,
            context_mask=context_mask,
        )
        tokens = self.mot(
            video_state=video_state,
            action_state=action_state,
            residual_injection=self.world_residual if apply_world_residual else None,
            sparse=sparse,
            step_index=step_index,
            num_steps=num_steps,
        )
        pred_video, pred_flow = self.video_expert.post_dit(
            tokens["video"],
            video_state,
        )
        result = {
            "video": pred_video,
            "flow": pred_flow,
            "action": self.action_expert.post_dit(tokens["action"]),
        }
        if predict_world_targets:
            result.update(
                self.world_residual.predict(
                    tokens=tokens["video"],
                    time_embedding=video_state["time_embedding"],
                    video_expert=self.video_expert,
                    grid_size=video_state["grid_size"],
                )
            )
            if self.config.local_preview_enabled:
                result["local_preview"] = self.world_residual.predict_local_previews(
                    time_embedding=video_state["time_embedding"],
                    video_expert=self.video_expert,
                    grid_size=video_state["grid_size"],
                )
        return result

    @staticmethod
    def _normalize_world(target: torch.Tensor) -> torch.Tensor:
        if target.ndim != 5:
            raise ValueError(
                f"world target must be [B,C,T,H,W], got {tuple(target.shape)}."
            )
        reduce_dims = tuple(range(1, target.ndim))
        mean = target.mean(dim=reduce_dims, keepdim=True)
        std = target.std(
            dim=reduce_dims,
            keepdim=True,
            unbiased=False,
        ).clamp_min(1.0e-6)
        return (target - mean) / std

    def _latent_valid_mask(
        self,
        image_is_pad: torch.Tensor | None,
        *,
        batch_size: int,
        loss_steps: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if image_is_pad is None:
            return None
        if image_is_pad.ndim != 2 or image_is_pad.shape[0] != batch_size:
            raise ValueError(
                f"image_is_pad must be [B,T], got {tuple(image_is_pad.shape)}."
            )
        temporal_factor = self.config.vae_temporal_downsample_factor
        if temporal_factor <= 0:
            raise ValueError("vae_temporal_downsample_factor must be positive.")
        if image_is_pad.shape[1] < 1 or (
            image_is_pad.shape[1] - 1
        ) % temporal_factor:
            raise ValueError(
                "image_is_pad cannot be aligned with Wan latent steps: "
                f"frames={image_is_pad.shape[1]}, factor={temporal_factor}."
            )
        latent_is_pad = image_is_pad[:, 1:].reshape(
            batch_size,
            -1,
            temporal_factor,
        ).all(dim=2)
        if latent_is_pad.shape[1] != loss_steps:
            raise ValueError(
                "image_is_pad latent mask length mismatch: "
                f"{latent_is_pad.shape[1]} vs {loss_steps}."
            )
        return (~latent_is_pad).to(device=device, dtype=dtype)

    def _future_mse(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        weight_map: torch.Tensor | None = None,
        image_is_pad: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError(
                f"prediction/target shape mismatch: {tuple(prediction.shape)} "
                f"vs {tuple(target.shape)}."
            )
        if prediction.ndim != 5 or prediction.shape[2] <= 1:
            raise ValueError("Future latent loss requires [B,C,T,H,W] with T > 1.")
        loss_map = F.mse_loss(
            prediction[:, :, 1:].float(),
            target[:, :, 1:].float(),
            reduction="none",
        ).mean(dim=1)
        valid = self._latent_valid_mask(
            image_is_pad,
            batch_size=prediction.shape[0],
            loss_steps=loss_map.shape[1],
            device=loss_map.device,
            dtype=loss_map.dtype,
        )
        if weight_map is None:
            per_step = loss_map.mean(dim=(2, 3))
            if valid is None:
                return per_step.mean(dim=1)
            denominator = valid.sum(dim=1).clamp_min(1.0)
            return (per_step * valid).sum(dim=1) / denominator
        if weight_map.shape != (prediction.shape[0], 1, *prediction.shape[2:]):
            raise ValueError(
                "weight_map must be [B,1,T,H,W], "
                f"got {tuple(weight_map.shape)} for {tuple(prediction.shape)}."
            )
        weight = weight_map[:, 0, 1:].detach().to(
            device=loss_map.device,
            dtype=loss_map.dtype,
        )
        if valid is not None:
            weight = weight * valid[:, :, None, None]
        denominator = weight.sum(dim=(1, 2, 3)).clamp_min(1.0)
        return (loss_map * weight).sum(dim=(1, 2, 3)) / denominator

    @staticmethod
    def _action_mse(
        prediction: torch.Tensor,
        target: torch.Tensor,
        action_is_pad: torch.Tensor | None,
    ) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError(
                f"action prediction/target mismatch: {tuple(prediction.shape)} "
                f"vs {tuple(target.shape)}."
            )
        token_loss = F.mse_loss(
            prediction.float(),
            target.float(),
            reduction="none",
        ).mean(dim=2)
        if action_is_pad is None:
            return token_loss.mean(dim=1)
        if action_is_pad.shape != prediction.shape[:2]:
            raise ValueError(
                "action_is_pad must match [B,T], "
                f"got {tuple(action_is_pad.shape)}."
            )
        valid = (~action_is_pad.bool()).to(dtype=token_loss.dtype)
        return (token_loss * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)

    @staticmethod
    def _require_training_batch(batch: dict) -> None:
        required = {
            "video_latents",
            "flow_latents",
            "dino_latents",
            "depth_latents",
            "action",
            "action_is_pad",
            "context",
            "context_mask",
            "image_is_pad",
            "proprio",
        }
        missing = sorted(required - batch.keys())
        if missing:
            raise KeyError(f"training batch is missing keys: {missing}")
        non_tensors = sorted(
            key for key in required if not isinstance(batch[key], torch.Tensor)
        )
        if non_tensors:
            raise TypeError(f"training batch values must be tensors: {non_tensors}")

    def training_loss(
        self,
        batch: dict[str, torch.Tensor],
        *,
        progress: float = 0.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not 0.0 <= progress <= 1.0:
            raise ValueError("progress must be in [0,1].")
        self._require_training_batch(batch)
        video = batch["video_latents"]
        flow = self._normalize_world(batch["flow_latents"])
        dino = self._normalize_world(batch["dino_latents"])
        depth = self._normalize_world(batch["depth_latents"])
        action = batch["action"]
        context = batch["context"]
        context_mask = batch["context_mask"]
        action_is_pad = batch["action_is_pad"]
        image_is_pad = batch["image_is_pad"]
        proprio = batch["proprio"]
        if proprio.ndim != 3 or proprio.shape[0] != video.shape[0]:
            raise ValueError(
                f"proprio must be [B,T,{self.config.proprio_dim}], "
                f"got {tuple(proprio.shape)}."
            )
        if proprio.shape[2] != self.config.proprio_dim:
            raise ValueError(
                f"proprio width must be {self.config.proprio_dim}, "
                f"got {proprio.shape[2]}."
            )

        batch_size = video.shape[0]
        if min(video.shape[2], flow.shape[2], dino.shape[2], depth.shape[2]) <= 1:
            raise ValueError("Training requires at least two latent time steps.")
        video_noise = torch.randn_like(video)
        video_timestep = self.video_scheduler.sample_training_t(
            batch_size,
            video.device,
            video.dtype,
        )
        flow_noise = torch.randn_like(flow)
        action_noise = torch.randn_like(action)
        action_timestep = self.action_scheduler.sample_training_t(
            batch_size,
            action.device,
            action.dtype,
        )
        noisy_video = self.video_scheduler.add_noise(
            video,
            video_noise,
            video_timestep,
        )
        noisy_flow = self.video_scheduler.add_noise(
            flow,
            flow_noise,
            video_timestep,
        )
        noisy_action = self.action_scheduler.add_noise(
            action,
            action_noise,
            action_timestep,
        )
        noisy_video[:, :, :1] = video[:, :, :1]
        noisy_flow[:, :, :1] = 0

        prediction = self(
            video_latents=noisy_video,
            flow_latents=noisy_flow,
            action_tokens=noisy_action,
            video_timestep=video_timestep,
            action_timestep=action_timestep,
            context=context,
            context_mask=context_mask,
            proprio=proprio[:, 0],
        )
        target_video = self.video_scheduler.training_target(video, video_noise)
        target_flow = self.video_scheduler.training_target(flow, flow_noise)
        target_action = self.action_scheduler.training_target(action, action_noise)

        video_per_sample = self._future_mse(
            prediction["video"],
            target_video,
            image_is_pad=image_is_pad,
        )
        flow_per_sample = self._future_mse(
            prediction["flow"],
            target_flow,
            image_is_pad=image_is_pad,
        )
        action_per_sample = self._action_mse(
            prediction["action"],
            target_action,
            action_is_pad,
        )
        video_weight = self.video_scheduler.training_weight(video_timestep)
        action_weight = self.action_scheduler.training_weight(action_timestep)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        dino_weight = self.config.dino_final_loss_weight + (
            self.config.dino_loss_weight - self.config.dino_final_loss_weight
        ) * cosine
        depth_weight = self.config.depth_final_loss_weight + (
            self.config.depth_loss_weight - self.config.depth_final_loss_weight
        ) * cosine
        dino_gate_map = self.world_residual.latest_gate_map(
            "dino",
            tuple(prediction["dino"].shape),
        )
        losses = {
            "video": (video_per_sample * video_weight).mean()
            * self.config.video_loss_weight,
            "flow": (flow_per_sample * video_weight).mean()
            * self.config.flow_loss_weight,
            "action": (action_per_sample * action_weight).mean()
            * self.config.action_loss_weight,
            "dino": self._future_mse(
                prediction["dino"],
                dino,
                weight_map=dino_gate_map,
                image_is_pad=image_is_pad,
            ).mean()
            * dino_weight,
            "depth": self._future_mse(
                prediction["depth"],
                depth,
                image_is_pad=image_is_pad,
            ).mean()
            * depth_weight,
        }
        local_preview_metrics = {}
        if self.config.local_preview_enabled:
            local_preview = prediction.get("local_preview")
            if not isinstance(local_preview, dict):
                raise RuntimeError("Parallel routing did not return local previews.")
            stream_targets = {"dino": dino, "depth": depth}
            stream_weights = {"dino": dino_weight, "depth": depth_weight}
            for name in self.world_residual.stream_names:
                records = local_preview.get(name)
                if not isinstance(records, tuple) or not records:
                    raise RuntimeError(f"No local-preview records for {name}.")
                layer_losses = []
                for layer_index, local_prediction in records:
                    layer_loss = self._future_mse(
                        local_prediction,
                        stream_targets[name],
                        image_is_pad=image_is_pad,
                    ).mean()
                    layer_losses.append(layer_loss)
                    local_preview_metrics[
                        f"{name}_local_preview_layer_{layer_index:02d}"
                    ] = layer_loss.detach()
                stacked = torch.stack(layer_losses)
                layer_weights = self.world_residual.local_preview_layer_weights(
                    len(layer_losses),
                    mode=self.config.local_preview_layer_weighting,
                    min_weight=self.config.local_preview_min_layer_weight,
                    device=stacked.device,
                    dtype=stacked.dtype,
                )
                local_raw = (stacked * layer_weights).sum() / layer_weights.sum()
                losses[f"{name}_local_preview"] = (
                    local_raw
                    * stream_weights[name]
                    * self.config.local_preview_loss_weight
                )
                local_preview_metrics[f"{name}_local_preview_raw"] = (
                    local_raw.detach()
                )
        gate_l1 = self.world_residual.gate_l1_loss()
        gate_l1_weighted = gate_l1 * self.config.gate_l1_weight
        total = torch.stack(tuple(losses.values())).sum() + gate_l1_weighted
        components = {name: value.detach() for name, value in losses.items()}
        components.update(local_preview_metrics)
        components["gate_l1"] = gate_l1.detach()
        components["gate_l1_weighted"] = gate_l1_weighted.detach()
        components["dino_weight"] = total.new_tensor(dino_weight)
        components["depth_weight"] = total.new_tensor(depth_weight)
        return total, components

    @torch.no_grad()
    def sample_action(
        self,
        *,
        first_frame_latents: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: torch.Tensor,
        action_horizon: int,
        num_video_latent_frames: int,
        num_steps: int = 10,
        seed: int | None = None,
        rand_device: str | torch.device = "cpu",
        sparse: SparseConfig | None = None,
    ) -> torch.Tensor:
        if first_frame_latents.ndim != 5 or first_frame_latents.shape[2] != 1:
            raise ValueError(
                "first_frame_latents must be [B,C,1,H,W], "
                f"got {tuple(first_frame_latents.shape)}."
            )
        if first_frame_latents.shape[1] != self.config.video_latent_dim:
            raise ValueError(
                f"expected {self.config.video_latent_dim} first-frame channels."
            )
        if action_horizon <= 0:
            raise ValueError("action_horizon must be positive.")
        if self.config.setting == "joint" and num_video_latent_frames <= 1:
            raise ValueError("Joint inference requires more than one video latent frame.")
        if num_steps <= 0:
            raise ValueError("num_steps must be positive.")
        if sparse is not None and sparse.enabled:
            if self.config.setting != "joint":
                raise ValueError(
                    "Sparse-WAM is implemented for the joint setting only; the "
                    f"'{self.config.setting}' path has no routed VV computation to restrict."
                )
            sparse.validate()
            self.mot.reset_sparse_diagnostics()

        batch, _, _, height, width = first_frame_latents.shape
        device = first_frame_latents.device
        dtype = first_frame_latents.dtype
        context = context.to(device=device, dtype=dtype)
        context_mask = context_mask.to(device=device, dtype=torch.bool)
        proprio = proprio.to(device=device, dtype=dtype)
        context, context_mask = self._append_proprio_to_context(
            context,
            context_mask,
            proprio,
        )

        random_device = torch.device(rand_device)
        action_generator = (
            None
            if seed is None
            else torch.Generator(device=random_device).manual_seed(seed)
        )
        action = torch.randn(
            batch,
            action_horizon,
            self.config.action_dim,
            device=random_device,
            dtype=torch.float32,
            generator=action_generator,
        ).to(device=device, dtype=dtype)
        action_steps, action_deltas = self.action_scheduler.inference_schedule(
            num_steps,
            device,
            dtype,
        )

        if self.config.setting == "uncond":
            flow = torch.zeros(
                batch,
                self.config.flow_latent_dim,
                1,
                height,
                width,
                device=device,
                dtype=dtype,
            )
            video_state = self.video_expert.pre_dit(
                video_latents=first_frame_latents,
                flow_latents=flow,
                timestep=torch.zeros(batch, device=device, dtype=dtype),
                context=context,
                context_mask=context_mask,
            )
            self.world_residual.reset_runtime()
            video_kv_cache = self.mot.prefill_video_cache(
                video_state=video_state,
                residual_injection=self.world_residual,
            )
            for action_step, action_delta in zip(action_steps, action_deltas):
                action_state = self.action_expert.pre_dit(
                    action_tokens=action,
                    timestep=action_step.expand(batch),
                    context=context,
                    context_mask=context_mask,
                )
                action_tokens = self.mot.forward_action_with_video_cache(
                    action_state=action_state,
                    video_kv_cache=video_kv_cache,
                    video_length=video_state["tokens"].shape[1],
                    video_tokens_per_frame=video_state["tokens_per_frame"],
                )
                action = self.action_scheduler.step(
                    self.action_expert.post_dit(action_tokens),
                    action_delta,
                    action,
                )
            return action

        video_generator = (
            None
            if seed is None
            else torch.Generator(device=random_device).manual_seed(seed)
        )
        video = torch.randn(
            batch,
            self.config.video_latent_dim,
            num_video_latent_frames,
            height,
            width,
            device=random_device,
            dtype=torch.float32,
            generator=video_generator,
        ).to(device=device, dtype=dtype)
        flow = torch.zeros(
            batch,
            self.config.flow_latent_dim,
            num_video_latent_frames,
            height,
            width,
            device=device,
            dtype=dtype,
        )
        video[:, :, :1] = first_frame_latents
        video_steps, video_deltas = self.video_scheduler.inference_schedule(
            num_steps,
            device,
            dtype,
        )
        for step_index, (video_step, video_delta, action_step, action_delta) in enumerate(
            zip(video_steps, video_deltas, action_steps, action_deltas)
        ):
            prediction = self._forward_conditioned(
                video_latents=video,
                flow_latents=flow,
                action_tokens=action,
                video_timestep=video_step.expand(batch),
                action_timestep=action_step.expand(batch),
                context=context,
                context_mask=context_mask,
                apply_world_residual=True,
                predict_world_targets=False,
                sparse=sparse,
                step_index=step_index,
                num_steps=num_steps,
            )
            video = self.video_scheduler.step(
                prediction["video"],
                video_delta,
                video,
            )
            action = self.action_scheduler.step(
                prediction["action"], action_delta, action
            )
            video[:, :, :1] = first_frame_latents
        return action


class DreamWAMUncond(DreamWAMJoint):
    def __init__(self, config: DreamWAMConfig = DreamWAMConfig(setting="uncond")):
        if config.setting != "uncond":
            config = replace(config, setting="uncond")
        super().__init__(config)
