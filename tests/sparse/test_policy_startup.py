"""Policy construction must work before any optional inference optimization."""

from types import SimpleNamespace

import pytest

import dreamwam.policy as policy_module


@pytest.mark.parametrize("fast_ops", [False, True])
def test_evaluation_is_available_when_optional_operators_are_selected(monkeypatch, fast_ops):
    calls = []
    model = SimpleNamespace(
        eval=lambda: None,
        enable_fast_ops=lambda **kwargs: calls.append(kwargs),
    )
    monkeypatch.setattr(policy_module, "build_model", lambda *args, **kwargs: model)
    monkeypatch.setattr(policy_module, "load_model_checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(policy_module, "load_wan_vae", lambda *args, **kwargs: object())
    monkeypatch.setattr(policy_module, "WanContextEncoder", lambda *args, **kwargs: object())
    monkeypatch.setattr(policy_module, "LiberoNormalizer", lambda *args, **kwargs: object())
    evaluation = dict(action_horizon=32, video_frames=33, denoising_steps=10,
                      seed=42, rand_device="cpu", binarize_gripper=True)
    if fast_ops:
        evaluation["fast_ops"] = True
    config = SimpleNamespace(
        evaluation=evaluation,
        paths=SimpleNamespace(checkpoint="unused", dataset_stats="unused"),
        preprocessing=dict(wan_vae_checkpoint="unused", wan_text_checkpoint="unused",
                           wan_tokenizer="unused", image_size=224),
    )
    policy = policy_module.DreamWAMPolicy(config, device="cpu")
    assert policy.evaluation is evaluation
    assert len(calls) == int(fast_ops)
