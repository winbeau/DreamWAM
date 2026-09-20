"""A requested visual option must reach execution and the worker fingerprint."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import dreamwam.policy as policy_module
from dreamwam.sparse.config import config_hash
from dreamwam.sparse.action_guided_visual_token_cache import ActionGuidedVisualTokenCache
from dreamwam.sparse.conditioned_frame_cache import ConditionedFrameCache
from dreamwam.sparse.fresh_visual_tokens import FreshVisualTokenSparsity
from test_visual_step_cache import model_and_inputs


@pytest.mark.parametrize("graph_mode,conditioned,fresh", [
    (None, False, False), ("dense_action", False, False), ("all_transformers", False, False),
    (None, True, False), ("all_transformers", True, False),
    (None, False, True), ("all_transformers", False, True),
])
def test_adapter_executes_guided_tokens_and_publishes_effective_options(monkeypatch, tmp_path, graph_mode, conditioned, fresh):
    if graph_mode and not torch.cuda.is_available():
        pytest.skip("actual graph/adapter execution requires CUDA on the evaluation server")
    device = "cuda" if graph_mode else "cpu"
    dtype = torch.bfloat16 if graph_mode else torch.float32
    # Match runtime.build_model's device context: complex RoPE tables are plain
    # attributes, deliberately excluded from Module.to(dtype=...) conversion.
    with torch.device(device):
        model, inputs = model_and_inputs()
    model.to(device=device, dtype=dtype)
    inputs = {key: value.to(device=device, dtype=dtype if value.is_floating_point() else value.dtype)
              if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}
    reference = (FreshVisualTokenSparsity(model, keep_ratio=0.25) if fresh else
                 ConditionedFrameCache(model, refresh_every=1) if conditioned else
                 ActionGuidedVisualTokenCache(model, keep_ratio=0.25, refresh_every=2))
    with reference:
        expected = model.sample_action(**inputs).squeeze(0).float().cpu().numpy()
    monkeypatch.setattr(policy_module, "build_model", lambda *args, **kwargs: model)
    monkeypatch.setattr(policy_module, "load_model_checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(policy_module, "load_wan_vae", lambda *args, **kwargs: object())
    class TextEncoder:
        def __init__(self):
            self.model = torch.nn.Identity().eval().requires_grad_(False)
            self.tokenizer = SimpleNamespace(seq_len=3, clean=None)
            self.device, self.dtype = torch.device(device), dtype
            self.calls = 0

        def __call__(self, prompts):
            self.calls += 1
            # Feed the returned tensors into real Joint sampling below.
            return inputs["context"].clone(), inputs["context_mask"].clone()
    encoder = TextEncoder()
    monkeypatch.setattr(policy_module, "WanContextEncoder", lambda *args, **kwargs: encoder)
    monkeypatch.setattr(policy_module, "LiberoNormalizer", lambda *args, **kwargs: object())
    # Keep real Joint sampling; heavyweight image/text encoders are outside this
    # wiring test and are covered by the checkpoint pilot.
    def predict_action(self, **kwargs):
        with torch.no_grad():
            context, mask = self.text_encoder([kwargs["instruction"]])
            return self.model.sample_action(**{**inputs, "context": context,
                                               "context_mask": mask}).squeeze(0).float().cpu().numpy()
    monkeypatch.setattr(policy_module.DreamWAMPolicy, "predict_action", predict_action)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"fixture")
    release_file = tmp_path / "release.yaml"
    release_file.write_text("fixture: true\n")
    release = SimpleNamespace(
        setting="joint",
        evaluation=dict(action_horizon=4, video_frames=9, denoising_steps=4,
                        seed=42, rand_device="cpu", binarize_gripper=True,
                        replan_steps=4, wait_steps=30),
        paths=SimpleNamespace(checkpoint=checkpoint, dataset_stats="unused"),
        preprocessing=dict(wan_vae_checkpoint="unused", wan_text_checkpoint="unused",
                           wan_tokenizer="unused", image_size=224),
    )
    adapter_file = Path(__file__).resolve().parents[2] / "evaluation/action_eval/infer.py"
    spec = importlib.util.spec_from_file_location("visual_adapter_fixture", adapter_file)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    monkeypatch.setattr(adapter, "_import_dreamwam", lambda root: {
        "load_release_config": lambda path: release,
        "build_policy": policy_module.DreamWAMPolicy,
        "config_hash": config_hash,
    })
    options = (dict(keep_ratio=0.25, selection="action") if fresh else
               dict(refresh_every=1, conditioned_frame_reuse=True) if conditioned else
               dict(refresh_every=2, token_keep_ratio=0.25, action_guidance_weight=1.0))
    if graph_mode:
        options["graph_dispatch"] = graph_mode
    option_key = "fresh_visual_tokens" if fresh else "visual_cache"
    policy = adapter.create_policy(dict(model_root=str(tmp_path), model_config=str(release_file),
                                        **{option_key: options}, prompt_cache={"capacity": 2}), device=device)
    cache = policy.policy._fresh_visual_runtime if fresh else policy.policy._visual_cache_runtime
    prompt_cache = policy.policy._prompt_cache_runtime
    try:
        assert policy.describe().fingerprint[option_key] == options
        assert policy.describe().fingerprint["prompt_cache"] == {"capacity": 2}
        policy.reset(dict(episode_id="fixture", seed=42))
        observation = dict(
            images={key: np.zeros((16, 16, 3), dtype=np.uint8) for key in ("agentview", "wrist")},
            state=np.zeros(8, dtype=np.float32), instruction="move the block",
        )
        prediction = policy.predict(observation)
        assert np.array_equal(prediction.actions, expected)
        assert prediction.diagnostics["prompt_cache"]["last_hit"] is False
        prediction = policy.predict(observation)
        assert np.array_equal(prediction.actions, expected)
        assert prediction.diagnostics["prompt_cache"]["last_hit"] is True
        assert encoder.calls == 1
        assert prediction.actions.shape == (4, 7)
        stats = prediction.diagnostics[option_key]
        if fresh:
            assert stats["denoising_steps"] == stats["selection_builds"] == 4
            assert stats["computed_video_token_layers"] == 24
            assert stats["reused_visual_steps"] == 0
            assert not cache._indices
        else:
            assert stats["dense_video_steps"] == 1
            assert stats["partial_video_steps"] == (3 if conditioned else 1)
            assert stats["reused_video_steps"] == (0 if conditioned else 2)
            assert stats.get("action_mass_builds", 0) == (0 if conditioned else 1)
            assert not policy.policy._visual_cache_runtime.video_kv
        assert stats["action_layer_updates"] == 8
        assert len(stats["selected_indices"][0]) == (8 if conditioned else 3)
        if graph_mode and fresh:
            assert stats["graph_replays"] == 4 and cache.graph_stats()["captured"]
        elif graph_mode:
            assert stats["dense_graph_replays"] == 1
            assert stats["action_graph_replays"] == (0 if conditioned else 2)
            assert stats["partial_graph_replays"] == (3 if conditioned else int(graph_mode == "all_transformers"))
            assert all(value["captured"] for value in cache.graph_stats().values())
        policy.predict({**observation, "instruction": "move the other block"})
        assert encoder.calls == 2 and not prompt_cache.last_hit
        policy.reset(dict(episode_id="next", seed=42))
        assert not prompt_cache.entries
        assert np.array_equal(policy.predict(observation).actions, expected)
        assert encoder.calls == 3 and not prompt_cache.last_hit
    finally:
        policy.close()
    assert not getattr(model, "_visual_ffn_context_cache", None)
    assert not prompt_cache.entries
    if graph_mode and fresh:
        assert cache.replay is None
    elif graph_mode:
        assert not cache.graphs
