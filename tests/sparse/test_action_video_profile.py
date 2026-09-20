import json

import numpy as np
import pytest
import torch

from dreamwam.sparse.profile.admission import admit_profile, eligible_devices
from dreamwam.sparse.profile.archive import RawArchive, sha256
from dreamwam.sparse.profile.capture import DenseProfile
from dreamwam.sparse.profile.geometry import TokenGrid
from dreamwam.sparse.profile.signals import joint_probabilities, token_signals, value_denoising_drift
from test_visual_step_cache import model_and_inputs


def test_real_grid_camera_mapping_and_ragged_regions():
    grid = TokenGrid(3, 7, 14)
    coordinates = grid.coordinates()
    assert coordinates[7].tolist() == [7, 0, 0, 7, 1, 0]
    assert coordinates[14].tolist() == [14, 0, 1, 0, 0, 0]
    assert coordinates[98].tolist() == [98, 1, 0, 0, 0, 0]
    regions = grid.regions()
    assert sorted(np.concatenate(regions).tolist()) == list(range(294))
    assert set(map(len, regions)) == {1, 2, 4}
    for group in regions:
        assert len(set(coordinates[group, 1])) == len(set(coordinates[group, 4])) == 1
    with pytest.raises(ValueError, match="seam"):
        TokenGrid(3, 7, 13)


def test_full_joint_denominator_value_norm_and_two_time_axes():
    grid = TokenGrid(3, 1, 2)
    q, k = torch.zeros(2, 8, 4), torch.zeros(2, 8, 4)
    v = torch.arange(8).reshape(1, 8, 1).expand(2, 8, 4).float()
    mask = torch.ones(8, 8, dtype=torch.bool)
    mask[:6, 6:] = False
    mask[:2, 2:6] = False
    p = joint_probabilities(q, k, mask)
    assert torch.equal(p[:, 6:, :], torch.full((2, 2, 8), 1/8))
    assert not p[:, :6, 6:].any()
    scores = token_signals(p, v, grid)
    assert torch.equal(scores["action"], torch.full((2, 6), 1/8))
    assert torch.equal(scores["value_action"], scores["value_norm"] / 8)
    assert scores["action_key_mass"].tolist() == [0.25, 0.25]
    assert scores["value_video_time"][0].tolist() == [0, 0, 4, 4, 8, 8]
    # Add a common denoising shift: video-time differences unchanged, drift nonzero.
    assert torch.equal(token_signals(p, v+10, grid)["value_video_time"], scores["value_video_time"])
    assert value_denoising_drift(v+10, v, 6).unique().tolist() == [20]
    mask[0] = False
    with pytest.raises(ValueError, match="visible"):
        joint_probabilities(q, k, mask)


def test_context_propagates_to_supporting_keys_not_consuming_queries():
    grid = TokenGrid(1, 1, 2)
    p = torch.tensor([[[1., 0., 0.], [1., 0., 0.], [0., 1., 0.]]])
    result = token_signals(p, torch.ones(1, 3, 2), grid)
    assert result["action"].tolist() == [[0., 1.]]
    assert result["action_context_support"].tolist() == [[1., 0.]]


def test_profile_preserves_native_outputs_replays_probabilities_and_isolates_requests():
    model, inputs = model_and_inputs()
    expected = model.sample_action(**inputs)
    records = []
    original = model.mot._joint_self_attention
    with DenseProfile(model, steps=(0, 1, 2, 3), layers=(0, 1), heads=(0, 1),
                      sink=lambda *args: records.append(args), max_records=8) as profile:
        actual = model.sample_action(**inputs)
        assert torch.equal(actual, expected)
        assert profile.stats["records"] == 8
        assert profile.stats["video_steps"] == 4
        assert not profile.previous
        for kind, metadata, arrays in records:
            if kind != "attention":
                continue
            p = joint_probabilities(*(torch.from_numpy(arrays[key]) for key in ("query", "key", "mask")))
            combined = np.concatenate((arrays["vv_probabilities"], arrays["av_probabilities"]), axis=1)
            assert np.array_equal(p.numpy(), combined)
            assert ("value_denoising_drift" in arrays) == (metadata["step"] > 0)
        first_values = next(a["value"].copy() for k, _, a in records if k == "attention")
        records.clear()
        changed = dict(inputs, first_frame_latents=inputs["first_frame_latents"] + 1)
        model.sample_action(**changed)
        attention = [(m, a) for k, m, a in records if k == "attention"]
        assert attention[0][0]["request"] == 2
        assert attention[0][0]["previous_sampled_step"] is None
        assert "value_denoising_drift" not in attention[0][1]
        assert not np.array_equal(attention[0][1]["value"], first_values)
    assert model.mot._joint_self_attention == original
    assert not getattr(model, "_visual_ffn_context_cache", None)
    assert torch.equal(model.sample_action(**inputs), expected)


def test_sampling_gaps_are_never_called_adjacent_drift():
    model, inputs = model_and_inputs()
    records = []
    with DenseProfile(model, steps=(0, 3), layers=(1,), heads=(1,), sink=lambda *args: records.append(args)):
        model.sample_action(**inputs)
    later = next(m for k, m, _ in records if k == "attention" and m["step"] == 3)
    assert later["previous_sampled_step"] == 0
    assert later["adjacent_denoising_step"] is False


@pytest.mark.parametrize("failure", ["sink", "bytes", "steps"])
def test_failures_restore_hooks_and_discard_request_state(failure):
    model, inputs = model_and_inputs()
    expected = model.sample_action(**inputs)
    original = model.mot.forward
    def sink(*args):
        if failure == "sink":
            raise OSError("disk full")
    profile = DenseProfile(model, steps=(10,) if failure == "steps" else (0,),
                           layers=(0,), heads=(0,), sink=sink,
                           max_bytes=1 if failure == "bytes" else 1024**2)
    with pytest.raises((OSError, RuntimeError, ValueError)):
        with profile:
            model.sample_action(**inputs)
    assert model.mot.forward == original
    assert not profile.previous and not profile.active
    assert profile.raw_action is profile.last_video is None
    assert torch.equal(model.sample_action(**inputs), expected)


def test_archive_refuses_overwrite_and_verifies_raw_replay(tmp_path):
    archive = RawArchive(tmp_path / "raw", max_bytes=100)
    arrays = dict(values=np.arange(4, dtype=np.float32))
    metadata = dict(request=1, step=0, layer=0)
    row = archive.write("t0", "attention", metadata, arrays)
    assert sha256(archive.root / row["path"]) == row["sha256"]
    with np.load(archive.root / row["path"], allow_pickle=False) as actual:
        assert np.array_equal(actual["values"], arrays["values"])
    assert json.loads((archive.root / "records.jsonl").read_text())["raw_bytes"] == 16
    with pytest.raises(FileExistsError):
        archive.write("t0", "attention", metadata, arrays)
    with pytest.raises(RuntimeError, match="budget"):
        archive.write("t1", "attention", metadata, dict(values=np.zeros(100)))
    assert len(archive.records) == 1


def test_admission_excludes_unauthorized_cards_and_busy_authorized_cards():
    text = "2, GPU-2, NVIDIA H100, 4, 81559, 0\n3, GPU-3, NVIDIA H100, 24000, 81559, 100\n4, GPU-4, NVIDIA H100, 24000, 81559, 82\n5, GPU-5, NVIDIA H100, 4, 81559, 0\n"
    assert [d["index"] for d in eligible_devices(text)] == [5]


def test_admission_never_consumes_last_available_card(monkeypatch):
    text = "3, GPU-3, NVIDIA H100, 24000, 81559, 100\n5, GPU-5, NVIDIA H100, 4, 81559, 0\n"
    monkeypatch.setattr("dreamwam.sparse.profile.admission.subprocess.check_output", lambda *a, **k: text)
    with pytest.raises(RuntimeError, match="last one unused"):
        admit_profile("GPU-5")
    text = "3, GPU-3, NVIDIA H100, 4, 81559, 0\n5, GPU-5, NVIDIA H100, 4, 81559, 0\n"
    assert admit_profile("GPU-3")["selected"] == "GPU-3"
    with pytest.raises(RuntimeError, match="not currently admitted"):
        admit_profile("GPU-2")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires separately admitted CUDA")
def test_actual_cuda_profile_preserves_native_eager():
    model, inputs = model_and_inputs()
    model = model.to("cuda")
    inputs = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    expected = model.sample_action(**inputs)
    with DenseProfile(model, steps=(0, 3), layers=(0, 1), heads=(0, 1), sink=lambda *args: None):
        assert torch.equal(model.sample_action(**inputs), expected)
