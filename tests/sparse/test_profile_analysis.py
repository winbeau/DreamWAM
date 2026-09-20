"""Replay independent raw arrays, including checksum-valid false evidence."""

import json

import numpy as np
import pytest

from dreamwam.sparse.profile.analysis import analyze_profile
from dreamwam.sparse.profile.archive import RawArchive, sha256
from dreamwam.sparse.profile.capture import DenseProfile
from dreamwam.sparse.profile.intervention import KeyIntervention
from dreamwam.sparse.profile.summary import summarize_analysis
from test_visual_step_cache import model_and_inputs


@pytest.fixture
def captured(tmp_path):
    source = tmp_path / "observations"
    source.mkdir()
    np.savez(source / "synthetic.npz", state=np.zeros(8, dtype=np.float32))
    input_sha = sha256(source / "synthetic.npz")
    manifest = source / "manifest.json"
    manifest.write_text(json.dumps(dict(status="CAPTURED", split_role="development",
        inputs=[dict(id="synthetic", path="synthetic.npz", sha256=input_sha)])))
    root = tmp_path / "capture"
    root.mkdir()
    archive = RawArchive(root / "raw", max_bytes=1024**2)
    model, inputs = model_and_inputs()
    with KeyIntervention(model, {}, operation="delete", scope="joint") as reference:
        expected = model.sample_action(**inputs)
        archive.write("synthetic", "reference", dict(request=1),
            dict(raw_action=reference.raw_action.numpy(), video_latents=reference.last_video.numpy()))
    with DenseProfile(model, steps=(0, 1, 2, 3), layers=(0, 1), heads=(0, 1),
        sink=lambda kind, metadata, arrays: archive.write("synthetic", kind, metadata, arrays)) as profile:
        actual = model.sample_action(**inputs)
        grid = [profile.grid.frames, profile.grid.height, profile.grid.width]
    archive.write("synthetic", "actions", dict(request=1),
                  dict(native=expected[0].numpy(), instrumented=actual[0].numpy()))
    report = dict(status="COMPLETE", exit_code=0, raw_schema_version=1,
        input_kind="synthetic_unit_test", source_commit="synthetic", checkpoint_sha256=None,
        split_role="development", input_manifest_sha256=sha256(manifest),
        sampling=dict(capture_steps=[0, 1, 2, 3], capture_layers=[0, 1], capture_heads=[0, 1],
            input_count=1, profile_inputs=str(manifest), profile_inputs_sha256=sha256(manifest),
            max_profile_raw_bytes=1024**2),
        inputs=[dict(input_id="synthetic", sha256=input_sha, grid=grid,
            action_parity=True, raw_action_parity=True, video_latent_parity=True)],
        completed_calls=2, planned_calls=2, artifacts=len(archive.records), raw_bytes=archive.raw_bytes,
        raw_index_sha256=sha256(root / "raw" / "records.jsonl"))
    (root / "report.json").write_text(json.dumps(report))
    return root


def test_independent_replay_exports_equal_budget_routes_and_typed_evidence(captured, tmp_path):
    out = tmp_path / "analysis"
    result = analyze_profile(captured, out, read_count=3)
    assert result["status"] == "COMPLETE" and result["attention_records"] == 8
    assert result["token_rows"] == 192 and result["validated_files"] == 19
    assert result["real_model_evidence"] is False and result["sr"] is None
    routes = [json.loads(line) for line in (out / "selections.jsonl").read_text().splitlines()]
    assert {r["method"] for r in routes} >= {"uniform", "random", "action", "value_video_time", "action_context_support"}
    assert all(len(set(r["indices"])) == 3 and r["frame_counts"] == [1, 1, 1] for r in routes)
    transitions = [json.loads(line) for line in (out / "stability.jsonl").read_text().splitlines()]
    assert {r["axis"] for r in transitions} == {"denoising_step", "layer"}
    assert all(r["jaccard"] == 1 for r in transitions if r["method"] == "uniform" and r["frame"] is None)
    summary = summarize_analysis(out, tmp_path / "summary")
    assert summary["input_count"] == 1 and summary["real_model_evidence"] is False
    assert all(row["mean"] == 1 for row in summary["early_first_layer_to_final_last_layer"] if row["method"] == "uniform")
    routes_path = out / "selections.jsonl"
    routes_path.write_text(routes_path.read_text() + "{}\n")
    with pytest.raises(ValueError, match="hash differs"):
        summarize_analysis(out, tmp_path / "tampered-summary")


@pytest.mark.parametrize("fault,match", [("parity", "parity fails"),
    ("denominator", "softmax replay"), ("coverage", "coverage differs"), ("bytes", "hash/size mismatch")])
def test_rejects_invalid_raw_evidence_even_with_complete_status(captured, tmp_path, fault, match):
    index = captured / "raw" / "records.jsonl"
    rows = [json.loads(line) for line in index.read_text().splitlines()]
    row = next(r for r in rows if r["kind"] == ("final" if fault == "parity" else "attention"))
    path = captured / "raw" / row["path"]
    if fault == "coverage":
        rows.remove(row)
    elif fault == "bytes":
        path.write_bytes(path.read_bytes()[:-1])
    else:
        with np.load(path, allow_pickle=False) as raw:
            arrays = {k: raw[k].copy() for k in raw.files}
        if fault == "parity":
            arrays["raw_action"].flat[0] += 1
        else:
            # A visual-only denominator must not masquerade as native joint AV.
            probabilities = arrays["av_probabilities"]
            nv = row["video_length"]
            probabilities[..., :nv] /= probabilities[..., :nv].sum(axis=-1, keepdims=True)
            probabilities[..., nv:] = 0
        np.savez_compressed(path, **arrays)
        row.update(sha256=sha256(path), stored_bytes=path.stat().st_size)
    index.write_text("".join(json.dumps(r) + "\n" for r in rows))
    report_path = captured / "report.json"
    report = json.loads(report_path.read_text())
    report.update(raw_index_sha256=sha256(index), artifacts=len(rows), raw_bytes=sum(r["raw_bytes"] for r in rows))
    report_path.write_text(json.dumps(report))
    out = tmp_path / "invalid-analysis"
    with pytest.raises(ValueError, match=match):
        analyze_profile(captured, out, read_count=3)
    assert json.loads((out / "report.json").read_text())["status"] == "ERROR"
