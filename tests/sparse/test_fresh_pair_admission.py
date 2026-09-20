"""Resource admission for the six-visible-GPU H100 deployment (no CUDA needed)."""
import importlib.util
from pathlib import Path
import sys
import types

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
