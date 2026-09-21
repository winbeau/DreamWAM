"""M1 observes only causal inputs and must actually change fixed M2/M3 work."""

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import dreamwam.policy as policy_module
from dreamwam.sparse.chunk_budget import BudgetLevel, ChunkBudgetConfig, ObservationBudget, rotation_distance
from dreamwam.sparse.config import config_hash
from dreamwam.sparse.hybrid import HybridConfig
from dreamwam.sparse.hybrid.runtime import HybridVisualRuntime
from test_visual_step_cache import model_and_inputs


OPTIONS = Path(__file__).resolve().parents[2] / "configs/sparse/m1-fixed-m2-m3.json"


def observation(brightness=0, position=0, gripper=0):
    state = np.zeros(8, np.float32)
    state[0], state[6] = position, gripper
    return dict(images={name: np.full((32, 32, 3), brightness, np.uint8)
                        for name in ("agentview", "wrist")}, state=state)


def advance(controller, obs):
    decision = controller.propose(**obs)
    controller.commit(decision)
    return controller.last_stats


def test_causal_history_changes_budgets_and_first_chunk_is_high():
    controller = ObservationBudget(ChunkBudgetConfig())
    levels = [advance(controller, observation())["level"] for _ in range(5)]
    assert levels == ["high", "high", "medium", "medium", "low"]
    changed = advance(controller, observation(brightness=255))
    assert changed["level"] == "high" and changed["features"]["rgb_change"] == 1
    assert changed["reason"] == "observable_change_upgrade"
    assert changed["history_length"] == 3
    controller.reset()
    first = advance(controller, observation(brightness=255))
    assert first["reason"] == "bootstrap_no_history"
    assert first["history_length"] == first["chunk_index"] == 0
    assert all(value == 0 for value in first["features"].values())


@pytest.mark.parametrize("change,feature", [
    (dict(position=0.2), "translation"), (dict(gripper=0.05), "gripper"),
    (dict(brightness=100), "rgb_change"),
])
def test_each_raw_observation_channel_can_upgrade(change, feature):
    controller = ObservationBudget(ChunkBudgetConfig(downshift_after=1))
    for _ in range(3):
        advance(controller, observation())
    result = advance(controller, observation(**change))
    assert result["level"] == "high"
    assert result["features"][feature] > 0


def test_historical_trend_detects_innovation_and_rotation_wrap_is_not_motion():
    controller = ObservationBudget(ChunkBudgetConfig())
    advance(controller, observation(position=0))
    advance(controller, observation(position=0.1))
    continued = controller.propose(**observation(position=0.2))
    stopped = controller.propose(**observation(position=0.1))
    assert continued.diagnostics["features"]["translation_innovation"] == pytest.approx(0)
    assert stopped.diagnostics["features"]["translation_innovation"] == pytest.approx(0.1)
    assert rotation_distance(np.array([np.pi, 0, 0]), np.array([-np.pi, 0, 0])) == pytest.approx(0)
    assert rotation_distance(np.zeros(3), np.array([0, 0, np.pi / 2])) == pytest.approx(np.pi / 2)


def test_proposals_are_transactional_and_history_owns_its_arrays():
    controller = ObservationBudget(ChunkBudgetConfig())
    obs = observation()
    first = controller.propose(**obs)
    discarded = controller.propose(**observation(brightness=255))
    assert not controller._history and controller.last_stats == {}
    controller.commit(first)
    obs["images"]["agentview"].fill(255)
    obs["state"].fill(100)
    assert controller.propose(**observation()).diagnostics["features"]["rgb_change"] == 0
    with pytest.raises(RuntimeError, match="stale"):
        controller.commit(discarded)
    controller.reset()
    with pytest.raises(RuntimeError, match="stale"):
        controller.commit(first)


def test_fixed_control_reports_features_without_changing_its_budget():
    controller = ObservationBudget(ChunkBudgetConfig(mode="fixed", fixed_level=0))
    for value in (0, 255, 0):
        result = advance(controller, observation(brightness=value))
        assert result["level"] == "low" and result["reason"] == "fixed_budget_control"
    assert result["features"]["rgb_change"] == 1


@pytest.mark.parametrize("options", [
    {"unknown": 1}, {"schema_version": True}, {"history_size": 1},
    {"thumbnail_size": 0}, {"downshift_after": True}, {"fixed_level": 3},
    {"thresholds": [1, 0.5]}, {"thresholds": [float("nan"), 2]},
    {"scales": {"rgb_change": 0.1}}, {"mode": "teacher"},
    {"levels": [{"name": "invalid", "query_ratio": 0.6, "read_ratio": 0.5}]},
])
def test_invalid_configuration_is_rejected(options):
    with pytest.raises(ValueError):
        ChunkBudgetConfig.from_mapping(options)


def test_invalid_observations_cannot_advance_history():
    controller = ObservationBudget(ChunkBudgetConfig())
    obs = observation()
    for invalid in (dict(images={}, state=obs["state"]),
                    dict(images=obs["images"], state=np.full(8, np.nan)),
                    dict(images=obs["images"], state=np.zeros(7)),
                    dict(images={key: value.astype(float) for key, value in obs["images"].items()}, state=obs["state"])):
        with pytest.raises(ValueError):
            controller.propose(**invalid)
    assert not controller._history
    with pytest.raises(TypeError):
        controller.propose(**obs, success=True)


def test_frozen_config_only_changes_budgets_and_hashes_every_rule():
    payload = json.loads(OPTIONS.read_text())
    base = HybridConfig.from_mapping(payload["hybrid_visual"])
    budget = ChunkBudgetConfig.from_mapping(payload["chunk_budget"])
    assert ChunkBudgetConfig.from_mapping(budget.describe()).policy_hash == budget.policy_hash
    assert replace(budget, downshift_after=1).policy_hash != budget.policy_hash
    for config in budget.hybrid_configs(base):
        assert config.schedule is base.schedule
        assert replace(config, recompute_ratio=base.recompute_ratio, read_ratio=base.read_ratio) == base
    with pytest.raises(ValueError, match="compact"):
        budget.hybrid_configs(None)


def install_adapter_fixture(monkeypatch, tmp_path, backend, *, chunk_caps=False, adaptive=False):
    device = "cuda" if backend == "cuda_graph" else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    with torch.device(device):
        model, inputs = model_and_inputs()
    model.to(device=device, dtype=dtype)
    inputs = {key: value.to(device=device, dtype=dtype if value.is_floating_point() else value.dtype)
              if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}
    monkeypatch.setattr(policy_module, "build_model", lambda *args, **kwargs: model)
    monkeypatch.setattr(policy_module, "load_model_checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(policy_module, "load_wan_vae", lambda *args, **kwargs: object())
    monkeypatch.setattr(policy_module, "WanContextEncoder", lambda *args, **kwargs:
                        lambda prompts: (inputs["context"].clone(), inputs["context_mask"].clone()))
    monkeypatch.setattr(policy_module, "LiberoNormalizer", lambda *args, **kwargs:
                        SimpleNamespace(normalize_state=lambda value: value, denormalize_action=lambda value: value.clone()))
    monkeypatch.setattr(policy_module.DreamWAMPolicy, "_encode_first_frame", lambda self, images:
                        inputs["first_frame_latents"] + float(images["agentview"].mean()) / 255)
    checkpoint, release_file = tmp_path / "checkpoint.pt", tmp_path / "release.yaml"
    checkpoint.write_bytes(b"fixture")
    release_file.write_text("fixture: true\n")
    release = SimpleNamespace(setting="joint", model=model.config,
        evaluation=dict(action_horizon=4, video_frames=9, denoising_steps=10,
                        seed=42, rand_device="cpu", binarize_gripper=False),
        paths=SimpleNamespace(checkpoint=checkpoint, dataset_stats="unused"),
        preprocessing=dict(wan_vae_checkpoint="unused", wan_text_checkpoint="unused",
                           wan_tokenizer="unused", image_size=224))
    spec = importlib.util.spec_from_file_location("m1_adapter_fixture",
        Path(__file__).resolve().parents[2] / "evaluation/action_eval/infer.py")
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    monkeypatch.setattr(adapter, "_import_dreamwam", lambda root: dict(
        load_release_config=lambda path: release, build_policy=policy_module.DreamWAMPolicy, config_hash=config_hash))
    options = json.loads(OPTIONS.read_text())
    options.pop("prompt_cache")
    options["hybrid_visual"]["execution"]["backend"] = backend
    options["hybrid_visual"]["diagnostics"]["level"] = "trace"
    options["chunk_budget"]["downshift_after"] = 1
    if chunk_caps:
        for level in options["chunk_budget"]["levels"]:
            level["chunk_extra_passes"] = level["query_ratio"]
    if adaptive:
        options["hybrid_visual"]["step_router"] = dict(kind="adaptive", sparse_drift=0,
            dense_drift=1e-9, extra_dense_budget=9)
    return adapter.create_policy(dict(model_root=str(tmp_path), model_config=str(release_file), **options), device), inputs


@pytest.mark.parametrize("backend", ["eager", "buffered", "cuda_graph"])
@pytest.mark.parametrize("chunk_caps,adaptive", [(False, False), (True, False), (True, True)])
def test_adapter_executes_each_budget_with_frozen_m2_m3_and_own_eager_parity(monkeypatch, tmp_path, backend, chunk_caps, adaptive):
    if backend == "cuda_graph" and not torch.cuda.is_available():
        pytest.skip("CUDA graph verification runs only on admitted evaluation GPU")
    adapter, inputs = install_adapter_fixture(monkeypatch, tmp_path, backend, chunk_caps=chunk_caps, adaptive=adaptive)
    policy = adapter.policy
    runtime, controller = policy._hybrid_visual_runtime, policy._chunk_budget_runtime
    fingerprint = adapter.describe().fingerprint
    observed = []
    model = policy.model
    try:
        for obs in [observation(), observation(), observation(), observation(brightness=255)]:
            prediction = adapter.predict(dict(**obs, instruction="move block"))
            stats = prediction.diagnostics
            selected = stats["chunk_budget"]["level_index"]
            observed.append(selected)
            level = controller.config.levels[selected]
            assert stats["chunk_budget"]["executed"]["query_ratio"] == level.query_ratio
            assert stats["chunk_budget"]["executed"]["read_ratio"] == level.read_ratio
            assert stats["hybrid_visual"]["action_layer_updates"] == 20
            work = stats["hybrid_visual"]
            if not adaptive:
                assert [s["effective_op"] for s in work["steps"]] == list(runtime.base_config.schedule.operations)
            sparse = next(s for s in work["steps"] if s["effective_op"] == "sparse")
            assert sparse["q_rows"] == int(np.ceil(12 * level.query_ratio))
            assert sparse["kv_rows"] == int(np.ceil(12 * level.read_ratio))
            if chunk_caps:
                assert work["query_row_cap"] == 12 + int(np.ceil(12 * level.query_ratio))
                assert work["query_rows_spent"] == work["query_row_cap"]
                assert work["query_rows_unused"] == 0
                assert work["dense_steps"] == work["sparse_steps"] == 1
                assert work["computed_video_token_layers"] == 2 * work["query_row_cap"]
            assert not runtime.state.kv and not runtime._active
            assert adapter.describe().fingerprint == fingerprint, "worker identity cannot change with an online budget"
            config = replace(runtime.config, backend="eager")
            runtime.__exit__(None, None, None)
            try:
                with HybridVisualRuntime(model, config), torch.no_grad():
                    expected = model.sample_action(**{**inputs, "num_steps": 10,
                        "first_frame_latents": policy._encode_first_frame(obs["images"]),
                        "proprio": torch.from_numpy(obs["state"]).unsqueeze(0).to(policy.device, policy.dtype)})
                expected = expected[0].float().cpu().numpy()
                expected[:, -1] = -(expected[:, -1] * 2 - 1)
                assert prediction.actions.tobytes() == expected.tobytes()
            finally:
                runtime.__enter__()
        assert observed == [2, 1, 0, 2]
        adapter.reset(dict(episode_id="new"))
        assert not controller._history
        repeated = adapter.predict(dict(**observation(), instruction="move block"))
        assert repeated.diagnostics["chunk_budget"]["reason"] == "bootstrap_no_history"
        assert repeated.diagnostics["chunk_budget"]["history_length"] == 0
        if backend == "cuda_graph":
            assert all(row["captured"] for row in runtime.graph_stats().values())
            assert len(runtime.dispatch.entries) <= runtime.base_config.max_graphs
    finally:
        adapter.close()
    assert not controller._history and not runtime.dispatch.entries
    assert not getattr(model, "_visual_ffn_context_cache", None)


def test_failed_policy_does_not_commit_observation(monkeypatch, tmp_path):
    adapter, _ = install_adapter_fixture(monkeypatch, tmp_path, "eager")
    policy = adapter.policy
    original = policy.model._forward_conditioned
    def fail(**kwargs):
        raise RuntimeError("injected sampling failure")
    monkeypatch.setattr(policy.model, "_forward_conditioned", fail)
    try:
        with pytest.raises(RuntimeError, match="injected"):
            adapter.predict(dict(**observation(), instruction="move block"))
        assert not policy._chunk_budget_runtime._history
        assert not policy._hybrid_visual_runtime.state.kv
        monkeypatch.setattr(policy.model, "_forward_conditioned", original)
        actual = adapter.predict(dict(**observation(), instruction="move block"))
        assert actual.diagnostics["chunk_budget"]["history_length"] == 0
    finally:
        adapter.close()


def test_chunk_allocations_are_canonical_and_distinct_from_per_refresh_quotas():
    levels = tuple(BudgetLevel(name, .1, .75, cap) for name, cap in
                   (("low", .1), ("medium", .4), ("high", .8)))
    config = ChunkBudgetConfig(levels=levels)
    assert ChunkBudgetConfig.from_mapping(config.describe()) == config
    base = HybridConfig.from_mapping(json.loads(OPTIONS.read_text())["hybrid_visual"])
    compiled = config.hybrid_configs(base)
    assert [c.query_row_cap(294, 10) for c in compiled] == [324, 412, 530]
    assert all(c.recompute_ratio == .1 and c.read_ratio == .75 for c in compiled)
    assert len({c.policy_hash for c in compiled}) == 3
    assert all(c.schedule is base.schedule for c in compiled)
    with pytest.raises(ValueError, match="all levels"):
        ChunkBudgetConfig(levels=(BudgetLevel("low", .1, .5), BudgetLevel("high", .2, .75, .5)))
    with pytest.raises(ValueError, match="monotonically"):
        ChunkBudgetConfig(levels=(BudgetLevel("low", .1, .5, .5), BudgetLevel("high", .2, .75, .1)), thresholds=(1,))
