from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .components import WanContextEncoder, load_wan_vae
from .config import ReleaseConfig
from .normalization import LiberoNormalizer
from .preprocessing.libero import PROMPT_TEMPLATE
from .runtime import build_model, load_model_checkpoint
from .sparse import SparseConfig
from .sparse.action_guided_visual_token_cache import ActionGuidedVisualTokenCache
from .sparse.conditioned_frame_cache import ConditionedFrameCache, GraphedConditionedFrameCache
from .sparse.chunk_budget import ChunkBudgetConfig, ObservationBudget
from .sparse.fresh_visual_tokens import FreshVisualTokenSparsity, fresh_visual_options
from .sparse.hybrid import HybridConfig
from .sparse.hybrid.runtime import HybridVisualRuntime
from .sparse.prompt_cache import PromptEncodingCache, prompt_cache_options
from .sparse.visual_cache_graphs import GraphedVisualTokenCache
from .sparse.visual_step_cache import VisualStepCache, visual_cache_options


def _center_crop_resize(image: np.ndarray, size: int) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
        raise ValueError(
            f"camera image must be uint8 [H,W,3], got {array.dtype} {array.shape}."
        )
    pil = Image.fromarray(array)
    source_width, source_height = pil.size
    scale = max(size / source_width, size / source_height)
    resized = pil.resize(
        (round(source_width * scale), round(source_height * scale)),
        resample=Image.Resampling.BILINEAR,
    )
    width, height = resized.size
    left = max((width - size) // 2, 0)
    top = max((height - size) // 2, 0)
    return np.asarray(
        resized.crop((left, top, left + size, top + size)),
        dtype=np.uint8,
    )


class DreamWAMPolicy:
    def __init__(
        self,
        config: ReleaseConfig,
        *,
        device: str | torch.device = "cuda",
        sparse: dict | None = None,
        visual_cache: dict | None = None,
        prompt_cache: dict | None = None,
        fresh_visual_tokens: dict | None = None,
        hybrid_visual: dict | None = None,
        chunk_budget: dict | None = None,
    ):
        self.config = config
        self.evaluation = config.evaluation
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA policy requested but no CUDA device is available.")
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        # Validated here rather than at first use, so a typo in an experiment YAML fails
        # the run at startup instead of silently producing a dense result.
        self.sparse_config = SparseConfig.from_mapping(sparse)
        self.visual_cache_config = visual_cache_options(visual_cache)
        self.prompt_cache_config = prompt_cache_options(prompt_cache)
        self.fresh_visual_config = fresh_visual_options(fresh_visual_tokens)
        hybrid_config = HybridConfig.from_mapping(hybrid_visual) if hybrid_visual is not None else None
        self.hybrid_visual_config = hybrid_config.describe() if hybrid_config is not None else None
        budget_config = ChunkBudgetConfig.from_mapping(chunk_budget) if chunk_budget is not None else None
        self.chunk_budget_config = budget_config.describe() if budget_config is not None else None
        self._chunk_budget_runtime = None
        if budget_config is not None:
            budget_config.hybrid_configs(hybrid_config)
            self._chunk_budget_runtime = ObservationBudget(budget_config)
        self._hybrid_visual_runtime = None
        self._visual_cache_runtime = None
        self._prompt_cache_runtime = None
        self._fresh_visual_runtime = None
        if hybrid_config is not None:
            if self.visual_cache_config is not None or self.fresh_visual_config is not None or self.sparse_config.enabled:
                raise ValueError("hybrid_visual must not be combined with another visual/sparse wrapper")
            if hybrid_config.backend == "cuda_graph" and self.device.type != "cuda":
                raise ValueError("hybrid_visual cuda_graph requires a CUDA device")
        if self.fresh_visual_config is not None:
            if self.visual_cache_config is not None or self.sparse_config.enabled:
                raise ValueError("fresh_visual_tokens must be measured without visual_cache or sparse attention")
            if self.fresh_visual_config.get("graph_dispatch") and self.device.type != "cuda":
                raise ValueError("fresh_visual_tokens graph dispatch requires a CUDA device")
        if self.visual_cache_config and "graph_dispatch" in self.visual_cache_config and self.device.type != "cuda":
            raise ValueError("visual-cache graph dispatch requires a CUDA device")
        if self.visual_cache_config is not None and self.sparse_config.enabled:
            raise ValueError("visual-cache and sparse-attention factors must be measured separately")
        self.model = build_model(
            config,
            device=self.device,
            dtype=self.dtype,
        )
        load_model_checkpoint(self.model, config.paths.checkpoint, strict=True)
        self.model.eval()
        # Explicit configuration, not a default: enabling fused operators changes the numerics
        # slightly, so it must be visible in the run's identity and separable in comparisons.
        if bool(self.evaluation.get("fast_ops", False)):
            self.model.enable_fast_ops(dtype=self.dtype)
        self.vae = load_wan_vae(
            config.preprocessing["wan_vae_checkpoint"],
            device=self.device,
            dtype=self.dtype,
        )
        self.text_encoder = WanContextEncoder(
            config.preprocessing["wan_text_checkpoint"],
            config.preprocessing["wan_tokenizer"],
            device=self.device,
            dtype=self.dtype,
            context_length=int(config.preprocessing.get("context_length", 128)),
        )
        if self.prompt_cache_config is not None:
            self._prompt_cache_runtime = PromptEncodingCache(self.text_encoder, **self.prompt_cache_config)
            self.text_encoder = self._prompt_cache_runtime
        self.normalizer = LiberoNormalizer(config.paths.dataset_stats)
        self.image_size = int(config.preprocessing["image_size"])
        required = {
            "action_horizon",
            "video_frames",
            "denoising_steps",
            "seed",
            "rand_device",
            "binarize_gripper",
        }
        missing = sorted(required - self.evaluation.keys())
        if missing:
            raise KeyError(f"Evaluation config is missing: {missing}")
        if self.visual_cache_config is not None:
            options = self.visual_cache_config
            if options.get("conditioned_frame_reuse"):
                cache_class = GraphedConditionedFrameCache if options.get("graph_dispatch") else ConditionedFrameCache
                self._visual_cache_runtime = cache_class(self.model, refresh_every=options["refresh_every"])
            elif "token_keep_ratio" in options:
                mode = options.get("graph_dispatch")
                cache_class = GraphedVisualTokenCache if mode else ActionGuidedVisualTokenCache
                graph_options = {"graph_partial": mode == "all_transformers"} if mode else {}
                self._visual_cache_runtime = cache_class(
                    self.model, refresh_every=options["refresh_every"],
                    keep_ratio=options["token_keep_ratio"],
                    guidance_weight=options["action_guidance_weight"],
                    **graph_options,
                )
            else:
                self._visual_cache_runtime = VisualStepCache(self.model, **options)
            self._visual_cache_runtime.__enter__()
        if self.fresh_visual_config is not None:
            options = self.fresh_visual_config
            self._fresh_visual_runtime = FreshVisualTokenSparsity(
                self.model, keep_ratio=options["keep_ratio"], selection=options["selection"],
                graph_enabled=bool(options.get("graph_dispatch")),
            )
            self._fresh_visual_runtime.__enter__()
        if hybrid_config is not None:
            self._hybrid_visual_runtime = HybridVisualRuntime(self.model, hybrid_config)
            self._hybrid_visual_runtime.__enter__()

    def validate_inference_options(self):
        """Validate after any adapter overrides, before text/VAE/model execution."""
        if self._hybrid_visual_runtime is not None:
            self._hybrid_visual_runtime.validate_num_steps(int(self.evaluation["denoising_steps"]))

    def reset(self) -> None:
        """Charge each episode its first instruction encoding; visual state is request-local."""
        if self._prompt_cache_runtime is not None:
            self._prompt_cache_runtime.clear()
        if self._hybrid_visual_runtime is not None:
            self._hybrid_visual_runtime.reset()
        if self._chunk_budget_runtime is not None:
            self._chunk_budget_runtime.reset()

    def close(self) -> None:
        """Remove inference wrappers before the adapter releases model modules."""
        if self._chunk_budget_runtime is not None:
            self._chunk_budget_runtime.reset()
        if self._hybrid_visual_runtime is not None:
            self._hybrid_visual_runtime.__exit__(None, None, None)
            self._hybrid_visual_runtime.close_graphs()
            self._hybrid_visual_runtime = None
        if self._fresh_visual_runtime is not None:
            self._fresh_visual_runtime.__exit__(None, None, None)
            self._fresh_visual_runtime.close_graphs()
            self._fresh_visual_runtime = None
        if self._visual_cache_runtime is not None:
            self._visual_cache_runtime.__exit__(None, None, None)
            if isinstance(self._visual_cache_runtime, GraphedVisualTokenCache):
                self._visual_cache_runtime.close_graphs()
            self._visual_cache_runtime = None
        if self._prompt_cache_runtime is not None:
            self._prompt_cache_runtime.clear()
            self.text_encoder = self._prompt_cache_runtime.encoder
            self._prompt_cache_runtime = None

    @torch.no_grad()
    def _encode_first_frame(self, images: dict[str, np.ndarray]) -> torch.Tensor:
        if set(images) != {"agentview", "wrist"}:
            raise KeyError("images must contain agentview and wrist exactly.")
        primary = _center_crop_resize(images["agentview"], self.image_size)
        wrist = _center_crop_resize(images["wrist"], self.image_size)
        combined = np.concatenate([primary, wrist], axis=1)
        tensor = (
            torch.from_numpy(combined.copy())
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device=self.device, dtype=self.dtype)
        )
        tensor = tensor.mul(2.0 / 255.0).sub(1.0).unsqueeze(2)
        latents = self.vae.single_encode(tensor, device=str(self.device))
        expected_shape = (
            1,
            self.config.model.video_latent_dim,
            1,
            self.image_size // 16,
            (2 * self.image_size) // 16,
        )
        if tuple(latents.shape) != expected_shape:
            raise ValueError(
                f"Wan VAE first-frame shape is {tuple(latents.shape)}, "
                f"expected {expected_shape}."
            )
        return latents

    @torch.no_grad()
    def predict_action(
        self,
        *,
        images: dict[str, np.ndarray],
        state: np.ndarray,
        instruction: str,
    ) -> np.ndarray:
        self.validate_inference_options()
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (self.config.model.proprio_dim,):
            raise ValueError(
                f"state must have shape ({self.config.model.proprio_dim},), "
                f"got {state.shape}."
            )
        if not np.isfinite(state).all():
            raise FloatingPointError("state contains non-finite values.")
        # Pure observation decision BEFORE text encoding, VAE and denoising. Its
        # bounded history is committed only after a valid action chunk is produced.
        budget_decision = None
        if self._chunk_budget_runtime is not None:
            budget_decision = self._chunk_budget_runtime.propose(images, state)
            level = self._chunk_budget_runtime.config.levels[budget_decision.level_index]
            self._hybrid_visual_runtime.set_budget(level.query_ratio, level.read_ratio)
        prompt = PROMPT_TEMPLATE.format(task=instruction)
        context, context_mask = self.text_encoder([prompt])
        proprio = self.normalizer.normalize_state(
            torch.from_numpy(state).unsqueeze(0)
        ).to(device=self.device, dtype=self.dtype)
        video_frames = int(self.evaluation["video_frames"])
        temporal_factor = self.config.model.vae_temporal_downsample_factor
        if video_frames <= 1 or (video_frames - 1) % temporal_factor:
            raise ValueError(
                "evaluation.video_frames must satisfy (T - 1) % "
                f"{temporal_factor} == 0."
            )
        action = self.model.sample_action(
            first_frame_latents=self._encode_first_frame(images),
            context=context,
            context_mask=context_mask,
            proprio=proprio,
            action_horizon=int(self.evaluation["action_horizon"]),
            num_video_latent_frames=(video_frames - 1) // temporal_factor + 1,
            num_steps=int(self.evaluation["denoising_steps"]),
            seed=int(self.evaluation["seed"]),
            rand_device=str(self.evaluation["rand_device"]),
            sparse=self.sparse_config,
        )
        action = self.normalizer.denormalize_action(action.float().cpu())[0]
        action[:, -1] = -(action[:, -1] * 2.0 - 1.0)
        if bool(self.evaluation["binarize_gripper"]):
            action[:, -1] = torch.sign(action[:, -1])
        if not torch.isfinite(action).all():
            raise FloatingPointError("DreamWAM produced non-finite actions.")
        if budget_decision is not None:
            self._chunk_budget_runtime.commit(budget_decision)
            actual = self._hybrid_visual_runtime.last_stats
            self._chunk_budget_runtime.last_stats["executed"] = {
                key: actual[key] for key in ("plan_hash", "base_policy_hash", "query_ratio", "read_ratio",
                    "denoising_steps", "dense_steps", "sparse_steps", "reuse_steps",
                    "computed_video_token_layers", "total_video_token_layers",
                    "read_video_token_layers", "action_probe_rows", "video_key_probe_rows",
                    "video_query_probe_rows", "action_layer_updates")}
        return action.numpy()


def build_policy(
    config: ReleaseConfig,
    *,
    device: str | torch.device = "cuda",
    sparse: dict | None = None,
    visual_cache: dict | None = None,
    prompt_cache: dict | None = None,
    fresh_visual_tokens: dict | None = None,
    hybrid_visual: dict | None = None,
    chunk_budget: dict | None = None,
) -> DreamWAMPolicy:
    checkpoint = Path(config.paths.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing DreamWAM checkpoint: {checkpoint}")
    return DreamWAMPolicy(config, device=device, sparse=sparse, visual_cache=visual_cache,
                         prompt_cache=prompt_cache, fresh_visual_tokens=fresh_visual_tokens,
                         hybrid_visual=hybrid_visual, chunk_budget=chunk_budget)
