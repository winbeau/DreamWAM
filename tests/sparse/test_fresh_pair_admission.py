"""Resource admission for the six-visible-GPU H100 deployment (no CUDA needed)."""
import importlib.util
from pathlib import Path
import sys
import types
import subprocess

import pytest


@pytest.fixture
def admission(monkeypatch):
    # The pure admission check does not load an experiment or import the evaluator.
    module = types.ModuleType("action_eval.config")
    module.load_experiment = None
    monkeypatch.setitem(sys.modules, "action_eval.config", module)
    path = Path(__file__).resolve().parents[2] / "scripts/sparse/run_fresh_token_pair.py"
    spec = importlib.util.spec_from_file_location("fresh_pair_admission", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner


def inventory():
    return [dict(index=i, uuid=f"gpu-{i}", util=0, used=0, free=81081, processes=[])
            for i in range(6)]


def test_h100_placement_leaves_authorized_spare(admission):
    assert admission.admit_placement(inventory(), "gpu-3", "gpu-4", [3, 4, 5]) == [5]


def test_empty_unauthorized_gpu_does_not_count_as_spare(admission):
    gpus = inventory()
    gpus[5]["util"] = 95
    assert admission.admit_placement(gpus, "gpu-3", "gpu-4", [3, 4, 5]) == []
    with pytest.raises(ValueError, match="outside"):
        admission.admit_placement(gpus, "gpu-2", "gpu-4", [3, 4, 5])


def test_missing_container_process_rows_do_not_mean_empty_renderer(admission):
    gpus = inventory()
    gpus[4]["used"] = 950
    assert admission.admit_placement(gpus, "gpu-3", "gpu-4", [3, 4, 5]) == []
    assert admission.admit_placement(gpus, "gpu-3", "gpu-4", [3, 4, 5], True) == []


def test_graphics_sharing_requires_explicit_flag_and_visible_graphics_only(admission):
    gpus = inventory()
    gpus[4].update(used=950, processes=[dict(pid=123, type="G")])
    assert admission.admit_placement(gpus, "gpu-3", "gpu-4", [3, 4, 5]) == []
    assert admission.admit_placement(gpus, "gpu-3", "gpu-4", [3, 4, 5], True) == [5]
    gpus[4]["processes"].append(dict(pid=456, type="C"))
    assert admission.admit_placement(gpus, "gpu-3", "gpu-4", [3, 4, 5], True) == []


def test_cpu_renderer_uses_only_policy_gpu_and_preserves_a_spare(admission):
    gpus = inventory()
    assert admission.admit_cpu_rendering(gpus, "gpu-3", [3, 4, 5]) == [4, 5]
    gpus[4]["util"] = gpus[5]["util"] = 95
    assert admission.admit_cpu_rendering(gpus, "gpu-3", [3, 4, 5]) == []
    with pytest.raises(ValueError, match="outside"):
        admission.admit_cpu_rendering(gpus, "gpu-2", [3, 4, 5])


def test_explicit_gpu5_sharing_allows_last_card_but_keeps_strict_headroom(admission):
    gpus = inventory()
    gpus[3]["util"] = gpus[4]["util"] = 100
    assert admission.admit_cpu_rendering(gpus, "gpu-5", [3, 4, 5]) == []
    assert admission.admit_shared_gpu5_cpu(gpus, "gpu-5", [3, 4, 5])
    gpus[5]["util"] = 11
    assert not admission.admit_shared_gpu5_cpu(gpus, "gpu-5", [3, 4, 5])
    gpus[5]["util"] = 50
    assert admission.admit_shared_gpu5_cpu(gpus, "gpu-5", [3, 4, 5], 50)
    gpus[5]["util"] = 51
    assert not admission.admit_shared_gpu5_cpu(gpus, "gpu-5", [3, 4, 5], 50)
    with pytest.raises(ValueError, match="10 or 50"):
        admission.admit_shared_gpu5_cpu(gpus, "gpu-5", [3, 4, 5], 100)
    gpus[5].update(util=0, free=49999)
    assert not admission.admit_shared_gpu5_cpu(gpus, "gpu-5", [3, 4, 5])
    assert not admission.admit_shared_gpu5_cpu(gpus, "gpu-5", [3, 4, 5], 50)
    with pytest.raises(ValueError, match="only for GPU 5"):
        admission.admit_shared_gpu5_cpu(gpus, "gpu-0", [0, 3, 4, 5])
    with pytest.raises(ValueError, match="only for GPU 5"):
        admission.admit_shared_gpu5_cpu(gpus, "gpu-5", [0, 3, 4])


def test_episode_budget_cli_refuses_oversized_pairs_before_any_model_or_evaluator_launch(admission, monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["launcher", "--eval-root", "/unused/eval", "--model-root", "/unused/model",
        "--out-dir", str(tmp_path / "out"), "--adapter-report", "/unused/report",
        "--dense-config", "/unused/dense", "--sparse-config", "/unused/sparse", "--planned-episodes", "26",
        "--episode-ledger", str(tmp_path / "ledger.json")])
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("over-budget pair started a process"))
    with pytest.raises(SystemExit) as error:
        admission.main()
    assert error.value.code == 2 and not (tmp_path / "out").exists()


def test_episode_evidence_keeps_errors_separate_from_unattempted_identities(admission, tmp_path):
    import json
    root = tmp_path / "arm"
    for number, (status, reason) in enumerate((("succeeded", "success"), ("error", "policy_error"),
                                              ("not_run", "not_attempted"))):
        path = root / "run/episodes" / str(number) / "attempts/0001/result.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(dict(status=status, termination_reason=reason)))
    evidence = admission.budget_evidence([dict(output=str(root))], "STOPPED_REQUIRES_REVIEW")
    assert evidence["recorded_attempts"] == 2 and evidence["recorded_not_run"] == 1
    assert not evidence["exact_attempt_count"] and len(evidence["terminal_artifacts"]) == 3


def test_explicit_pilot_checks_count_protocol_and_common_options(admission):
    from copy import deepcopy
    from types import SimpleNamespace

    def item():
        return SimpleNamespace(config=SimpleNamespace(
            benchmark=SimpleNamespace(planned_episodes=3),
            runtime=SimpleNamespace(policy_gpu_uuids=("gpu-3",), render_gpu_uuid=None)),
            resolved=dict(benchmark={"tasks": [0, 1, 2]}, protocol={"max_steps": 400},
                runtime={"policy_gpu_uuids": ["gpu-3"], "render_gpu_uuid": None},
                policy={"repo_root": "/frozen/model", "options": {
                    "action_horizon": 32, "denoising_steps": 10, "rng_mode": "fixed_per_predict",
                    "prompt_cache": {"capacity": 8}}}))
    pair = {"dense": item(), "sparse": item()}
    admission.validate_pair_configs(pair, 3, "gpu-3", None)
    with pytest.raises(ValueError, match="episode count"):
        admission.validate_pair_configs(pair, 50, "gpu-3", None)
    for section, key, value in (("protocol", "max_steps", 200), ("benchmark", "tasks", [1, 2, 3])):
        changed = deepcopy(pair)
        changed["sparse"].resolved[section][key] = value
        with pytest.raises(ValueError, match=section):
            admission.validate_pair_configs(changed, 3, "gpu-3", None)
    changed = deepcopy(pair)
    changed["sparse"].resolved["policy"]["options"]["denoising_steps"] = 5
    with pytest.raises(ValueError, match="common model option"):
        admission.validate_pair_configs(changed, 3, "gpu-3", None)


def test_hybrid_fingerprint_rejects_wrong_options_and_checkpoint(admission):
    from dreamwam.sparse.hybrid import HybridConfig
    from test_hybrid_schedule import options as hybrid_options
    options = dict(action_horizon=32, denoising_steps=10, rng_mode="fixed_per_predict",
                   prompt_cache={"capacity": 8}, hybrid_visual=hybrid_options())
    fingerprint = dict(options, hybrid_visual=HybridConfig.from_mapping(options["hybrid_visual"]).describe(),
                       checkpoint_sha256="verified-checkpoint")
    assert admission.fingerprint_matches(fingerprint, options, "verified-checkpoint")
    assert not admission.fingerprint_matches(fingerprint, options, "another-checkpoint")
    assert not admission.fingerprint_matches(fingerprint, dict(options, hybrid_visual={}), "verified-checkpoint")


def test_hybrid_fingerprint_normalizes_new_routing_defaults_without_weakening_checks(admission):
    from copy import deepcopy
    from dreamwam.sparse.hybrid import HybridConfig
    from test_hybrid_schedule import options as hybrid_options
    hybrid = hybrid_options()
    hybrid.update(selection=dict(method="action_context"), reuse=dict(mode="features"))
    options = dict(action_horizon=32, denoising_steps=10, rng_mode="fixed_per_predict",
                   prompt_cache={"capacity": 8}, hybrid_visual=hybrid)
    fingerprint = dict(options, hybrid_visual=HybridConfig.from_mapping(hybrid).describe(), checkpoint_sha256="x")
    assert admission.fingerprint_matches(fingerprint, options, "x")
    changed = deepcopy(options)
    changed["hybrid_visual"]["selection"]["context_weight"] = 2
    assert not admission.fingerprint_matches(fingerprint, changed, "x")


def test_cleanup_bounds_an_unresponsive_worker_without_touching_an_unrelated_process(admission):
    code = "import signal,time; signal.signal(signal.SIGINT, signal.SIG_IGN); print('ready',flush=True); time.sleep(60)"
    owned = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE,
                             text=True, start_new_session=True)
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                                  start_new_session=True)
    try:
        assert owned.stdout.readline().strip() == "ready"
        assert admission.stop_owned_process(owned, grace_seconds=0.05)
        assert owned.returncode is not None and unrelated.poll() is None
    finally:
        if owned.poll() is None:
            owned.kill()
        owned.wait()
        unrelated.terminate()
        unrelated.wait()
