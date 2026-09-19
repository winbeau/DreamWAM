from collections.abc import Callable

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .experts import ActionDiT, VideoDiT
from .layers import apply_rope, modulate, scaled_dot_product_attention
from .sparse import SparseConfig, build_token_layout, sparse_joint_attention


class JointMoT(nn.Module):
    def __init__(
        self,
        video_expert: VideoDiT,
        action_expert: ActionDiT,
        *,
        action_attends_all_video: bool,
        use_gradient_checkpointing: bool,
    ):
        super().__init__()
        if video_expert.num_layers != action_expert.num_layers:
            raise ValueError("VideoDiT and ActionDiT must have the same number of layers.")
        if video_expert.num_heads != action_expert.num_heads:
            raise ValueError("VideoDiT and ActionDiT must have the same number of heads.")
        if video_expert.attn_head_dim != action_expert.attn_head_dim:
            raise ValueError("VideoDiT and ActionDiT must have the same attention head dim.")
        self.video_expert = video_expert
        self.action_expert = action_expert
        self.num_layers = video_expert.num_layers
        self.num_heads = video_expert.num_heads
        self.action_attends_all_video = bool(action_attends_all_video)
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)
        #: Sparse-WAM inference state.  Populated per request; empty when sparse is off.
        self._layout_cache: dict[tuple, object] = {}
        self.sparse_diagnostics: dict[str, float] = {}

    def reset_sparse_diagnostics(self) -> None:
        self.sparse_diagnostics = {"calls": 0.0, "density_sum": 0.0, "fallback_sum": 0.0}

    def _token_layout(self, *, video_length: int, tokens_per_frame: int, block_size: int):
        key = (video_length, tokens_per_frame, block_size)
        layout = self._layout_cache.get(key)
        if layout is None:
            layout = build_token_layout(
                video_length=video_length,
                tokens_per_frame=tokens_per_frame,
                block_size=block_size,
                device=self.video_expert.blocks[0].self_attn.q.weight.device,
            )
            self._layout_cache = {key: layout}
        return layout

    def _joint_self_attention(
        self,
        *,
        video_io: tuple,
        action_io: tuple,
        attention_mask: torch.Tensor,
        video_length: int,
        tokens_per_frame: int,
        sparse: SparseConfig,
        step_index: int,
        num_steps: int,
    ) -> torch.Tensor:
        """Dense joint self-attention, or the Sparse-WAM VV-restricted equivalent."""
        if sparse is None or not sparse.enabled:
            return scaled_dot_product_attention(
                torch.cat([video_io[0], action_io[0]], dim=1),
                torch.cat([video_io[1], action_io[1]], dim=1),
                torch.cat([video_io[2], action_io[2]], dim=1),
                self.num_heads,
                attention_mask,
            )
        layout = self._token_layout(
            video_length=video_length,
            tokens_per_frame=tokens_per_frame,
            block_size=sparse.block_size,
        )
        video_output, action_output, stats = sparse_joint_attention(
            query_video=video_io[0],
            key_video=video_io[1],
            value_video=video_io[2],
            query_action=action_io[0],
            key_action=action_io[1],
            value_action=action_io[2],
            num_heads=self.num_heads,
            layout=layout,
            config=sparse,
            step_index=step_index,
            num_steps=num_steps,
            action_mask=attention_mask[video_length:],
        )
        diagnostics = self.sparse_diagnostics
        diagnostics["calls"] = diagnostics.get("calls", 0.0) + 1
        diagnostics["density_sum"] = diagnostics.get("density_sum", 0.0) + stats["density"]
        diagnostics["fallback_sum"] = (
            diagnostics.get("fallback_sum", 0.0) + stats["fallback_fraction"]
        )
        diagnostics["last"] = stats
        return torch.cat([video_output, action_output], dim=1)

    @staticmethod
    def _split_modulation(
        block: nn.Module,
        time_modulation: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        has_token_time = time_modulation.ndim == 4
        chunk_dim = 2 if has_token_time else 1
        modulation = block.modulation.to(
            device=time_modulation.device,
            dtype=time_modulation.dtype,
        )
        values = (modulation + time_modulation).chunk(6, dim=chunk_dim)
        if has_token_time:
            values = tuple(value.squeeze(2) for value in values)
        return values

    @classmethod
    def _attention_input(
        cls,
        block: nn.Module,
        tokens: torch.Tensor,
        frequencies: torch.Tensor,
        time_modulation: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        (
            shift_attention,
            scale_attention,
            gate_attention,
            shift_ffn,
            scale_ffn,
            gate_ffn,
        ) = cls._split_modulation(block, time_modulation)
        attention_input = modulate(
            block.norm1(tokens),
            shift_attention,
            scale_attention,
        )
        query = apply_rope(
            block.self_attn.norm_q(block.self_attn.q(attention_input)),
            frequencies,
            block.num_heads,
        )
        key = apply_rope(
            block.self_attn.norm_k(block.self_attn.k(attention_input)),
            frequencies,
            block.num_heads,
        )
        value = block.self_attn.v(attention_input)
        return (
            query,
            key,
            value,
            tokens,
            gate_attention,
            shift_ffn,
            scale_ffn,
            gate_ffn,
        )

    @staticmethod
    def _post_attention(
        block: nn.Module,
        residual: torch.Tensor,
        mixed_attention: torch.Tensor,
        gate_attention: torch.Tensor,
        shift_ffn: torch.Tensor,
        scale_ffn: torch.Tensor,
        gate_ffn: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        tokens = residual + gate_attention * block.self_attn.o(mixed_attention)
        tokens = tokens + block.cross_attn(
            block.norm3(tokens),
            context,
            context_mask,
        )
        ffn_input = modulate(block.norm2(tokens), shift_ffn, scale_ffn)
        return tokens + gate_ffn * block.ffn(ffn_input)

    def build_attention_mask(
        self,
        *,
        video_length: int,
        action_length: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        if action_length <= 0:
            raise ValueError("action_length must be positive.")
        total_length = video_length + action_length
        mask = torch.zeros(
            total_length,
            total_length,
            dtype=torch.bool,
            device=device,
        )
        mask[:video_length, :video_length] = self.video_expert.build_attention_mask(
            video_length,
            video_tokens_per_frame,
            device,
        )
        if self.action_attends_all_video:
            mask[video_length:, :video_length] = True
        else:
            first_frame_length = min(video_tokens_per_frame, video_length)
            mask[video_length:, :first_frame_length] = True
        mask[video_length:, video_length:] = True
        return mask

    def prefill_video_cache(
        self,
        *,
        video_state: dict,
        residual_injection: Callable[
            [int, torch.Tensor, torch.Tensor, torch.Tensor],
            torch.Tensor,
        ]
        | None,
    ) -> list[dict[str, torch.Tensor]]:
        video_tokens = video_state["tokens"]
        video_attention_mask = self.video_expert.build_attention_mask(
            video_tokens.shape[1],
            video_state["tokens_per_frame"],
            video_tokens.device,
        )
        cache = []
        for layer_index in range(self.num_layers):
            video_block = self.video_expert.blocks[layer_index]
            video_io = self._attention_input(
                video_block,
                video_tokens,
                video_state["freqs"],
                video_state["time_modulation"],
            )
            mixed_attention = scaled_dot_product_attention(
                video_io[0],
                video_io[1],
                video_io[2],
                self.num_heads,
                video_attention_mask,
            )
            video_tokens = self._post_attention(
                video_block,
                video_io[3],
                mixed_attention,
                *video_io[4:],
                video_state["context"],
                video_state["context_mask"],
            )
            if residual_injection is not None:
                video_tokens = residual_injection(
                    layer_index,
                    video_tokens,
                    video_state["context"],
                    video_state["context_mask"],
                )
            cache.append({"k": video_io[1], "v": video_io[2]})
        return cache

    def forward_action_with_video_cache(
        self,
        *,
        action_state: dict,
        video_kv_cache: list[dict[str, torch.Tensor]],
        video_length: int,
        video_tokens_per_frame: int,
    ) -> torch.Tensor:
        if len(video_kv_cache) != self.num_layers:
            raise ValueError(
                f"video_kv_cache must have {self.num_layers} layers, "
                f"got {len(video_kv_cache)}."
            )
        action_tokens = action_state["tokens"]
        attention_mask = self.build_attention_mask(
            video_length=video_length,
            action_length=action_tokens.shape[1],
            video_tokens_per_frame=video_tokens_per_frame,
            device=action_tokens.device,
        )
        action_attention_mask = attention_mask[video_length:]

        for layer_index in range(self.num_layers):
            action_block = self.action_expert.blocks[layer_index]
            action_io = self._attention_input(
                action_block,
                action_tokens,
                action_state["freqs"],
                action_state["time_modulation"],
            )
            layer_cache = video_kv_cache[layer_index]
            if set(layer_cache) != {"k", "v"}:
                raise ValueError(
                    f"video_kv_cache[{layer_index}] must contain k and v exactly."
                )
            video_key = layer_cache["k"]
            video_value = layer_cache["v"]
            if video_key.shape[1] != video_length or video_value.shape[1] != video_length:
                raise ValueError(
                    f"video_kv_cache[{layer_index}] sequence length mismatch."
                )
            mixed_attention = scaled_dot_product_attention(
                action_io[0],
                torch.cat([video_key, action_io[1]], dim=1),
                torch.cat([video_value, action_io[2]], dim=1),
                self.num_heads,
                action_attention_mask,
            )
            action_tokens = self._post_attention(
                action_block,
                action_io[3],
                mixed_attention,
                *action_io[4:],
                action_state["context"],
                action_state["context_mask"],
            )
        return action_tokens

    def forward(
        self,
        *,
        video_state: dict,
        action_state: dict,
        residual_injection: Callable[
            [int, torch.Tensor, torch.Tensor, torch.Tensor],
            torch.Tensor,
        ]
        | None,
        sparse: SparseConfig | None = None,
        step_index: int = 0,
        num_steps: int = 1,
    ) -> dict[str, torch.Tensor]:
        video_tokens = video_state["tokens"]
        action_tokens = action_state["tokens"]
        attention_mask = self.build_attention_mask(
            video_length=video_tokens.shape[1],
            action_length=action_tokens.shape[1],
            video_tokens_per_frame=video_state["tokens_per_frame"],
            device=video_tokens.device,
        )
        tokens_per_frame = int(video_state["tokens_per_frame"])

        for layer_index in range(self.num_layers):
            video_block = self.video_expert.blocks[layer_index]
            action_block = self.action_expert.blocks[layer_index]

            def forward_layer(
                current_video: torch.Tensor,
                current_action: torch.Tensor,
                video_block: nn.Module = video_block,
                action_block: nn.Module = action_block,
                layer_index: int = layer_index,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                video_io = self._attention_input(
                    video_block,
                    current_video,
                    video_state["freqs"],
                    video_state["time_modulation"],
                )
                action_io = self._attention_input(
                    action_block,
                    current_action,
                    action_state["freqs"],
                    action_state["time_modulation"],
                )
                video_length = current_video.shape[1]
                mixed_attention = self._joint_self_attention(
                    video_io=video_io,
                    action_io=action_io,
                    attention_mask=attention_mask,
                    video_length=video_length,
                    tokens_per_frame=tokens_per_frame,
                    sparse=sparse,
                    step_index=step_index,
                    num_steps=num_steps,
                )
                next_video = self._post_attention(
                    video_block,
                    video_io[3],
                    mixed_attention[:, :video_length],
                    *video_io[4:],
                    video_state["context"],
                    video_state["context_mask"],
                )
                next_action = self._post_attention(
                    action_block,
                    action_io[3],
                    mixed_attention[:, video_length:],
                    *action_io[4:],
                    action_state["context"],
                    action_state["context_mask"],
                )
                if residual_injection is not None:
                    next_video = residual_injection(
                        layer_index,
                        next_video,
                        video_state["context"],
                        video_state["context_mask"],
                    )
                return next_video, next_action

            if self.training and self.use_gradient_checkpointing:
                video_tokens, action_tokens = checkpoint(
                    forward_layer,
                    video_tokens,
                    action_tokens,
                    use_reentrant=False,
                )
            else:
                video_tokens, action_tokens = forward_layer(
                    video_tokens,
                    action_tokens,
                )

        return {"video": video_tokens, "action": action_tokens}
