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
from test_visual_step_cache import model_and_inputs


@pytest.mark.parametrize("graph_mode", [None, "dense_action", "all_transformers"])
def test_adapter_executes_guided_tokens_and_publishes_effective_options(monkeypatch, tmp_path, graph_mode):
    if graph_mode and not torch.cuda.is_available():
        pytest.skip("actual graph/adapter execution requires CUDA on the evaluation server")
    device = "cuda" if graph_mode else "cpu"
    dtype = torch.bfloat16 if graph_mode else torch.float32
    model, inputs = model_and_inputs()
    model.to(device=device, dtype=dtype)
    inputs = {key: value.to(device=device, dtype=dtype if value.is_floating_point() else value.dtype)
              if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}
    with ActionGuidedVisualTokenCache(model, keep_ratio=0.25, refresh_every=2):
        expected = model.sample_action(**inputs).squeeze(0).float().cpu().numpy()
    monkeypatch.setattr(policy_module, "build_model", lambda *args, **kwargs: model)
    monkeypatch.setattr(policy_module, "load_model_checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(policy_module, "load_wan_vae", lambda *args, **kwargs: object())
    monkeypatch.setattr(policy_module, "WanContextEncoder", lambda *args, **kwargs: object())
    monkeypatch.setattr(policy_module, "LiberoNormalizer", lambda *args, **kwargs: object())
    # Keep real Joint sampling; heavyweight image/text encoders are outside this
    # wiring test and are covered by the checkpoint pilot.
    def predict_action(self, **kwargs):
        with torch.no_grad():
            return self.model.sample_action(**inputs).squeeze(0).float().cpu().numpy()
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
    options = dict(refresh_every=2, token_keep_ratio=0.25, action_guidance_weight=1.0)
    if graph_mode:
        options["graph_dispatch"] = graph_mode
    policy = adapter.create_policy(dict(model_root=str(tmp_path), model_config=str(release_file),
                                        visual_cache=options), device=device)
    cache = policy.policy._visual_cache_runtime
    try:
        assert policy.describe().fingerprint["visual_cache"] == options
        policy.reset(dict(episode_id="fixture", seed=42))
        observation = dict(
            images={key: np.zeros((16, 16, 3), dtype=np.uint8) for key in ("agentview", "wrist")},
            state=np.zeros(8, dtype=np.float32), instruction="move the block",
        )
        prediction = policy.predict(observation)
        assert np.array_equal(prediction.actions, expected)
        prediction = policy.predict(observation)
        assert np.array_equal(prediction.actions, expected)
        assert prediction.actions.shape == (4, 7)
        stats = prediction.diagnostics["visual_cache"]
        assert stats["dense_video_steps"] == 1
        assert stats["partial_video_steps"] == 1
        assert stats["reused_video_steps"] == 2
        assert stats["action_mass_builds"] == 1
        assert stats["action_layer_updates"] == 8
        assert len(stats["selected_indices"][0]) == 3
        assert not policy.policy._visual_cache_runtime.video_kv
        if graph_mode:
            assert stats["dense_graph_replays"] == 1
            assert stats["action_graph_replays"] == 2
            assert stats["partial_graph_replays"] == int(graph_mode == "all_transformers")
            assert all(value["captured"] for value in cache.graph_stats().values())
    finally:
        policy.close()
    assert not getattr(model, "_visual_ffn_context_cache", None)
    if graph_mode:
        assert not cache.graphs
