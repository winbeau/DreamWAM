"""Exercise visual refresh through real Joint sampling and transformer layers."""

import pytest
import torch

from dreamwam.model import DreamWAMConfig, DreamWAMJoint
from dreamwam.sparse.visual_step_cache import VisualStepCache, visual_cache_options


def model_and_inputs():
    model = DreamWAMJoint(DreamWAMConfig(
        video_latent_dim=4, flow_latent_dim=4, dino_latent_dim=2, depth_latent_dim=2,
        text_dim=8, freq_dim=8, video_hidden_dim=16, video_ffn_dim=32,
        action_hidden_dim=16, action_ffn_dim=32, num_heads=2, attn_head_dim=8,
        num_layers=2, dino_injection_layers=(0, 1), depth_injection_layers=(0, 1),
        use_gradient_checkpointing=False,
    )).eval()
    inputs = dict(first_frame_latents=torch.randn(1, 4, 1, 4, 4),
                  context=torch.randn(1, 3, 8), context_mask=torch.ones(1, 3, dtype=torch.bool),
                  proprio=torch.randn(1, 8), action_horizon=4,
                  num_video_latent_frames=3, num_steps=4, seed=42)
    return model, inputs


def test_refresh_every_step_is_bitwise_native_dense():
    model, inputs = model_and_inputs()
    dense = model.sample_action(**inputs)
    original = model.mot.forward
    with VisualStepCache(model, refresh_every=1) as cache:
        actual = model.sample_action(**inputs)
        assert torch.equal(actual, dense)
        assert cache.last_stats == dict(dense_video_steps=4, reused_video_steps=0,
                                       video_layer_updates=8, action_layer_updates=8)
        assert not cache.video_kv and cache.video_output is None
    assert model.mot.forward == original


def test_visual_reuse_keeps_every_action_step_and_both_schedulers(monkeypatch):
    model, inputs = model_and_inputs()
    calls = dict(video_layers=0, action_layers=0, video_scheduler=0, action_scheduler=0)
    handles = []
    for expert, key in ((model.video_expert, "video_layers"), (model.action_expert, "action_layers")):
        for block in expert.blocks:
            def hook(module, args, result, name=key):
                calls[name] += 1
            handles.append(block.self_attn.q.register_forward_hook(hook))
    for scheduler, key in ((model.video_scheduler, "video_scheduler"), (model.action_scheduler, "action_scheduler")):
        original = scheduler.step
        def step(*args, _original=original, _key=key, **kwargs):
            calls[_key] += 1
            return _original(*args, **kwargs)
        monkeypatch.setattr(scheduler, "step", step)
    with VisualStepCache(model, refresh_every=2) as cache:
        actual = model.sample_action(**inputs)
        assert torch.isfinite(actual).all()
        assert calls == dict(video_layers=4, action_layers=8, video_scheduler=4, action_scheduler=4)
        assert cache.last_stats["dense_video_steps"] == 2
        assert cache.last_stats["reused_video_steps"] == 2
        assert torch.equal(model.sample_action(**inputs), actual), "request-local state must be reset"
    for handle in handles:
        handle.remove()


def test_failed_sampling_discards_visual_tensors():
    model, inputs = model_and_inputs()
    with VisualStepCache(model, refresh_every=5) as cache:
        invalid = {**inputs, "num_steps": 0}
        with pytest.raises(ValueError):
            model.sample_action(**invalid)
        assert not cache.video_kv and not cache._active
        assert torch.isfinite(model.sample_action(**inputs)).all()


@pytest.mark.parametrize("interval", [0, -1, True, 1.5])
def test_invalid_refresh_interval(interval):
    model, _ = model_and_inputs()
    with pytest.raises(ValueError):
        VisualStepCache(model, refresh_every=interval)


@pytest.mark.parametrize("options,dense_steps,guidance_builds", [
    ({"refresh_every": 2}, 2, 0),
    ({"refresh_every": 2, "token_keep_ratio": 0.25}, 1, 0),
    ({"refresh_every": 2, "token_keep_ratio": 0.25, "action_guidance_weight": 1.0}, 1, 1),
    ({"refresh_every": 1, "token_keep_ratio": 1.0, "action_guidance_weight": 1.0}, 4, 0),
])
def test_policy_option_installs_real_sampling_cache_and_close_restores_dense(
    monkeypatch, options, dense_steps, guidance_builds,
):
    from types import SimpleNamespace
    import dreamwam.policy as policy_module

    model, inputs = model_and_inputs()
    dense = model.sample_action(**inputs)
    monkeypatch.setattr(policy_module, "build_model", lambda *args, **kwargs: model)
    monkeypatch.setattr(policy_module, "load_model_checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(policy_module, "load_wan_vae", lambda *args, **kwargs: object())
    monkeypatch.setattr(policy_module, "WanContextEncoder", lambda *args, **kwargs: object())
    monkeypatch.setattr(policy_module, "LiberoNormalizer", lambda *args, **kwargs: object())
    config = SimpleNamespace(
        evaluation=dict(action_horizon=32, video_frames=9, denoising_steps=10,
                        seed=42, rand_device="cpu", binarize_gripper=True),
        paths=SimpleNamespace(checkpoint="unused", dataset_stats="unused"),
        preprocessing=dict(wan_vae_checkpoint="unused", wan_text_checkpoint="unused",
                           wan_tokenizer="unused", image_size=224),
    )
    policy = policy_module.DreamWAMPolicy(config, device="cpu", visual_cache=options)
    result = policy.model.sample_action(**inputs)
    stats = policy._visual_cache_runtime.last_stats
    assert stats["dense_video_steps"] == dense_steps
    assert stats["action_layer_updates"] == 8
    assert stats.get("action_mass_builds", 0) == guidance_builds
    assert stats.get("partial_video_steps", 0) == int(dense_steps == 1)
    assert policy.visual_cache_config == visual_cache_options(options)
    if options["refresh_every"] == 1:
        assert torch.equal(result, dense)
    policy.close()
    assert policy._visual_cache_runtime is None
    assert torch.equal(policy.model.sample_action(**inputs), dense)


def test_visual_option_defaults_preserve_temporal_identity_and_disable_implicit_guidance():
    assert visual_cache_options(None) is None
    assert visual_cache_options({"refresh_every": 5}) == {"refresh_every": 5}
    assert visual_cache_options({"refresh_every": 5, "token_keep_ratio": 0.1}) == {
        "refresh_every": 5, "token_keep_ratio": 0.1, "action_guidance_weight": 0.0,
    }


@pytest.mark.parametrize("payload", [
    {}, [], {"refresh_every": True}, {"refresh_every": 1.5}, {"refresh_every": 0},
    {"refresh_every": 5, "keep_ratio": 0.1},
    {"refresh_every": 5, "action_guidance_weight": 1.0},
    *({"refresh_every": 5, "token_keep_ratio": value}
      for value in (True, "0.1", None, -0.1, 1.1, float("nan"), float("inf"))),
    *({"refresh_every": 5, "token_keep_ratio": 0.1, "action_guidance_weight": value}
      for value in (True, "1", None, -1, float("nan"), float("inf"))),
])
def test_invalid_visual_options_fail_before_loading_weights(monkeypatch, payload):
    import dreamwam.policy as policy_module
    from types import SimpleNamespace

    def forbidden_load(*args, **kwargs):
        pytest.fail("invalid options reached model loading")

    monkeypatch.setattr(policy_module, "build_model", forbidden_load)
    with pytest.raises(ValueError, match="visual_cache"):
        policy_module.DreamWAMPolicy(SimpleNamespace(evaluation={}), device="cpu", visual_cache=payload)
