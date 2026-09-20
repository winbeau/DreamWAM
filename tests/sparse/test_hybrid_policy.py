"""Actual adapter -> policy -> Joint sampling, with only heavyweight encoders stubbed."""

from dataclasses import replace
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import dreamwam.policy as policy_module
from dreamwam.sparse.config import config_hash
from dreamwam.sparse.hybrid.runtime import HybridVisualRuntime
from test_hybrid_compact import compact
from test_visual_step_cache import model_and_inputs


@pytest.mark.parametrize("backend", ["eager", "buffered"])
def test_adapter_overrides_are_validated_then_executed_and_fingerprinted(monkeypatch, tmp_path, backend):
    model, inputs = model_and_inputs()
    conf = replace(compact(selection="uniform"), backend=backend)
    with HybridVisualRuntime(model, conf):
        expected = model.sample_action(**inputs).squeeze(0).numpy()
    monkeypatch.setattr(policy_module, "build_model", lambda *args, **kwargs: model)
    monkeypatch.setattr(policy_module, "load_model_checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(policy_module, "load_wan_vae", lambda *args, **kwargs: object())
    monkeypatch.setattr(policy_module, "WanContextEncoder", lambda *args, **kwargs: object())
    monkeypatch.setattr(policy_module, "LiberoNormalizer", lambda *args, **kwargs: object())
    def predict(self, **kwargs):
        self.validate_inference_options()
        return self.model.sample_action(**{**inputs, "num_steps": self.evaluation["denoising_steps"]}).squeeze(0).numpy()
    monkeypatch.setattr(policy_module.DreamWAMPolicy, "predict_action", predict)
    checkpoint, release_path = tmp_path / "checkpoint.pt", tmp_path / "release.yaml"
    checkpoint.write_bytes(b"fixture")
    release_path.write_text("fixture: true\n")
    release = SimpleNamespace(setting="joint",
        evaluation=dict(action_horizon=4, video_frames=9, denoising_steps=10,
                        seed=42, rand_device="cpu", binarize_gripper=True),
        paths=SimpleNamespace(checkpoint=checkpoint, dataset_stats="unused"),
        preprocessing=dict(wan_vae_checkpoint="unused", wan_text_checkpoint="unused",
                           wan_tokenizer="unused", image_size=224))
    adapter_path = Path(__file__).resolve().parents[2] / "evaluation/action_eval/infer.py"
    spec = importlib.util.spec_from_file_location("hybrid_adapter_fixture", adapter_path)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    monkeypatch.setattr(adapter, "_import_dreamwam", lambda root: dict(
        load_release_config=lambda path: release, build_policy=policy_module.DreamWAMPolicy,
        config_hash=config_hash))
    options = dict(model_root=str(tmp_path), model_config=str(release_path),
                   hybrid_visual=conf.describe(), denoising_steps=4)
    policy = adapter.create_policy(options, "cpu")
    runtime = policy.policy._hybrid_visual_runtime
    try:
        fingerprint = policy.describe().fingerprint
        assert fingerprint["hybrid_visual"] == conf.describe()
        assert fingerprint["hybrid_plan_hash"] == conf.policy_hash
        assert fingerprint["denoising_steps"] == 4
        observation = dict(images={key: np.zeros((16, 16, 3), np.uint8) for key in ("agentview", "wrist")},
                           state=np.zeros(8, np.float32), instruction="move block")
        prediction = policy.predict(observation)
        assert np.array_equal(prediction.actions, expected)
        stats = prediction.diagnostics["hybrid_visual"]
        assert stats["plan_hash"] == conf.policy_hash and stats["denoising_steps"] == 4
        assert [row["effective_op"] for row in stats["steps"]] == list(conf.schedule.operations)
        policy.reset(dict(episode_id="new"))
        assert not runtime.state.kv
    finally:
        policy.close()
    assert not runtime.dispatch.entries and not getattr(model, "_visual_ffn_context_cache", None)
    # The override is incompatible with the frozen schedule; no wrapper may remain.
    with pytest.raises(ValueError, match="num_steps"):
        adapter.create_policy({**options, "denoising_steps": 3}, "cpu")
    assert not getattr(model, "_visual_ffn_context_cache", None)


def test_conflicting_wrappers_rejected_before_loading_weights(monkeypatch):
    calls = []
    monkeypatch.setattr(policy_module, "build_model", lambda *args, **kwargs: calls.append(True))
    release = SimpleNamespace(evaluation={})
    with pytest.raises(ValueError, match="another"):
        policy_module.DreamWAMPolicy(release, device="cpu", hybrid_visual=compact().describe(),
                                     visual_cache=dict(refresh_every=5))
    assert not calls
