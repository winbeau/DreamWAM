"""Independent raw-profile replay and descriptive, equal-budget comparisons.

Checks actual arrays and declared coverage before a report can be COMPLETE.
No teacher signals, task outcomes, parameter fitting or GPU calls are involved.
"""

from contextlib import ExitStack
import csv
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import torch

from .archive import sha256
from .geometry import TokenGrid
from .selection import select_tokens
from .signals import joint_probabilities, token_signals, value_denoising_drift


SCORES = ("action", "value_norm", "value_action", "value_video_time",
          "visual_context", "action_context_support")


def stamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def jaccard(before, after):
    before, after = set(before), set(after)
    union = before | after
    return len(before & after) / len(union) if union else None


def load_record(raw_root, record):
    path = raw_root / record["path"]
    if Path(record["path"]).name != record["path"] or path.resolve().parent != raw_root.resolve():
        raise ValueError("raw artifact escapes its archive")
    if not path.is_file() or path.stat().st_size != record["stored_bytes"] or sha256(path) != record["sha256"]:
        raise ValueError(f"raw artifact hash/size mismatch: {path.name}")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    shapes = {name: dict(shape=list(value.shape), dtype=str(value.dtype)) for name, value in arrays.items()}
    if shapes != record["arrays"] or sum(a.nbytes for a in arrays.values()) != record["raw_bytes"]:
        raise ValueError("raw array schema/byte count differs from the index")
    if any(not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("raw archive contains nonfinite values")
    return arrays


def index_capture(capture_root):
    capture = json.loads((capture_root / "report.json").read_text())
    if capture.get("status") != "COMPLETE" or capture.get("exit_code") != 0:
        raise ValueError("incomplete/error capture cannot be analyzed as complete evidence")
    if capture.get("raw_schema_version") != 1:
        raise ValueError("unsupported raw profile schema")
    if capture.get("input_kind") not in ("self_captured_observations", "synthetic_unit_test"):
        raise ValueError("raw data provenance must be explicitly typed")
    sampling = capture["sampling"]
    steps, layers, heads = (sampling["capture_" + key] for key in ("steps", "layers", "heads"))
    for values in (steps, layers, heads):
        if not values or len(set(values)) != len(values) or any(type(v) is not int or v < 0 for v in values):
            raise ValueError("invalid sampling coverage")
    entries = capture["inputs"]
    ids = [entry["input_id"] for entry in entries]
    if len(ids) != sampling["input_count"] or len(set(ids)) != len(ids) or not ids:
        raise ValueError("input coverage incomplete or duplicated")
    if capture["completed_calls"] != capture["planned_calls"] or capture["planned_calls"] != 2 * len(ids):
        raise ValueError("policy call coverage differs from the declared paired capture")
    if any(entry.get(key) is not True for entry in entries
           for key in ("action_parity", "raw_action_parity", "video_latent_parity")):
        raise ValueError("capture parity flags did not pass")
    # Validate the original observation manifest rather than trusting its name.
    inputs_path = Path(sampling["profile_inputs"])
    if sha256(inputs_path) != capture["input_manifest_sha256"] or sha256(inputs_path) != sampling["profile_inputs_sha256"]:
        raise ValueError("original input manifest hash mismatch")
    input_manifest = json.loads(inputs_path.read_text())
    if input_manifest.get("status") != "CAPTURED" or input_manifest.get("split_role") != capture["split_role"]:
        raise ValueError("original input status or split differs from capture")
    original = input_manifest["inputs"]
    if len(original) != len(ids) or len({item["id"] for item in original}) != len(ids):
        raise ValueError("original observation coverage differs from capture")
    hashes = {item["id"]: item["sha256"] for item in original}
    if hashes != {item["input_id"]: item["sha256"] for item in entries}:
        raise ValueError("capture observations differ from the original manifest")
    for item in original:
        if sha256(inputs_path.parent / item["path"]) != item["sha256"]:
            raise ValueError("original observation bytes changed")
    raw_root = capture_root / "raw"
    if sha256(raw_root / "records.jsonl") != capture["raw_index_sha256"]:
        raise ValueError("raw index hash mismatch")
    records = [json.loads(line) for line in (raw_root / "records.jsonl").read_text().splitlines()]
    expected = {(name, kind, None, None) for name in ids for kind in ("reference", "final", "actions")}
    expected |= {(name, kind, step, None) for name in ids for step in steps for kind in ("step", "latent")}
    expected |= {(name, "attention", step, layer) for name in ids for step in steps for layer in layers}
    indexed = {}
    for row in records:
        identity = (row["input_id"], row["kind"], row.get("step"), row.get("layer"))
        if identity in indexed or row["request"] != 1:
            raise ValueError("duplicated or cross-request raw record")
        indexed[identity] = row
    paths = {row["path"] for row in records}
    if set(indexed) != expected or len(records) != capture["artifacts"]:
        raise ValueError("raw step/layer/input coverage differs from the declared profile")
    if len(paths) != len(records) or paths != {p.name for p in raw_root.glob("*.npz")}:
        raise ValueError("raw artifact set has duplicates, missing or unindexed files")
    if sum(row["raw_bytes"] for row in records) != capture["raw_bytes"]:
        raise ValueError("raw byte accounting differs from capture")
    if capture["raw_bytes"] > sampling["max_profile_raw_bytes"]:
        raise ValueError("raw data exceeds its predeclared byte budget")
    return capture, indexed


def analyze_profile(capture_root, out_dir, *, read_count=56):
    capture_root, out_dir = Path(capture_root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    report = dict(status="RUNNING", start_utc=stamp(), capture_root=str(capture_root.resolve()),
                  analysis_source_commit=subprocess.check_output(["git", "-C", str(Path(__file__).resolve().parents[3]),
                      "rev-parse", "HEAD"], text=True).strip(),
                  argv=sys.argv, python=platform.python_version(), torch=torch.__version__,
                  capture_report_sha256=sha256(capture_root / "report.json"), read_count=read_count,
                  token_rows=0, attention_records=0, selections=0, head_comparisons=0,
                  transitions=0, sr=None, speedup=None, semantic_annotation_coverage=None)
    write_json(out_dir / "report.json", report)
    try:
        capture, indexed = index_capture(capture_root)
        report.update(source_commit=capture["source_commit"], checkpoint_sha256=capture["checkpoint_sha256"],
            input_kind=capture["input_kind"], split_role=capture["split_role"], sampling=capture["sampling"],
            real_model_evidence=capture["input_kind"] == "self_captured_observations",
            raw_index_sha256=capture["raw_index_sha256"], validated_files=0,
            limitations=["attention/value/dynamic proxies are not causal importance or semantic labels",
                "selected heads and layers are a fixed sample, not exhaustive coverage",
                "all routes use equal balanced frame quotas; observed-frame video-time score is zero by construction",
                "empty/empty regional supports have undefined Jaccard, not perfect agreement",
                "no fusion weights, thresholds or online schedules fitted by this analysis"])
        with ExitStack() as stack:
            token_stream = stack.enter_context((out_dir / "token-scores.csv").open("w", newline=""))
            writer = csv.writer(token_stream)
            writer.writerow(["input_id", "step", "layer", "head", "token", "frame", "row", "column",
                             "camera", "camera_column", *SCORES, "value_denoising_drift"])
            supports = stack.enter_context((out_dir / "selections.jsonl").open("w"))
            stability = stack.enter_context((out_dir / "stability.jsonl").open("w"))
            head_stream = stack.enter_context((out_dir / "head-agreement.jsonl").open("w"))
            for entry in capture["inputs"]:
                name = entry["input_id"]
                grid = TokenGrid(*entry["grid"])
                select_tokens(grid, read_count, method="uniform")  # validates the common budget
                previous_values, previous_routes, previous_layer_routes = {}, {}, {}
                saved = {}
                for kind in ("reference", "final", "actions"):
                    saved[kind] = load_record(capture_root / "raw", indexed[name, kind, None, None])
                    report["validated_files"] += 1
                for key in ("raw_action", "video_latents"):
                    if not np.array_equal(saved["reference"][key], saved["final"][key]):
                        raise ValueError(f"raw {key} parity fails despite capture status")
                if not np.array_equal(saved["actions"]["native"], saved["actions"]["instrumented"]):
                    raise ValueError("executable action parity fails despite capture status")
                action_length = saved["final"]["raw_action"].shape[1]
                if saved["actions"]["native"].shape[0] != action_length:
                    raise ValueError("raw and executable action horizons differ")
                for step in sorted(capture["sampling"]["capture_steps"]):
                    step_data = load_record(capture_root / "raw", indexed[name, "step", step, None])
                    latent = load_record(capture_root / "raw", indexed[name, "latent", step, None])
                    report["validated_files"] += 2
                    if not np.array_equal(step_data["coordinates"], grid.coordinates()):
                        raise ValueError("raw frame/camera/space positions differ from the actual grid")
                    if latent["video_latents"].shape != saved["final"]["video_latents"].shape:
                        raise ValueError("latent layout changed across denoising steps")
                    for layer in sorted(capture["sampling"]["capture_layers"]):
                        row = indexed[name, "attention", step, layer]
                        arrays = load_record(capture_root / "raw", row)
                        report["validated_files"] += 1
                        if (row["heads"] != capture["sampling"]["capture_heads"] or
                            TokenGrid(**row["grid"]) != grid or row["video_length"] != grid.length or
                            row["action_length"] != action_length or row["qk_position"] != "native_post_rope" or
                            row["denominator"] != "all_native_visual_and_action_keys_with_original_mask"):
                            raise ValueError("attention metadata/layout differs from the native capture contract")
                        query, key, value, mask = (torch.from_numpy(arrays[k]) for k in ("query", "key", "value", "mask"))
                        if query.shape[:2] != (len(row["heads"]), grid.length + action_length) or query.shape != key.shape or key.shape != value.shape:
                            raise ValueError("raw projections omit a query, key or head")
                        p = joint_probabilities(query, key, mask)
                        stored = np.concatenate((arrays["vv_probabilities"], arrays["av_probabilities"]), axis=1)
                        if not np.array_equal(p.numpy(), stored):
                            raise ValueError("full joint softmax replay differs from saved probabilities")
                        scores = token_signals(p, value, grid)
                        for method, score in scores.items():
                            if not np.array_equal(score.numpy(), arrays["score_" + method]):
                                raise ValueError("score replay differs from raw projections")
                        prior = previous_values.get(layer)
                        drift = None
                        if row["previous_sampled_step"] != (None if prior is None else prior[0]):
                            raise ValueError("denoising drift references another request or step")
                        if row["adjacent_denoising_step"] != (prior is not None and prior[0] == step - 1):
                            raise ValueError("sampled denoising gap mislabeled as adjacent")
                        if prior is not None:
                            drift = value_denoising_drift(value, prior[1], grid.length).numpy()
                            if not np.array_equal(drift, arrays["value_denoising_drift"]):
                                raise ValueError("denoising drift replay differs from raw values")
                        elif "value_denoising_drift" in arrays:
                            raise ValueError("first sampled step contains stale denoising drift")
                        previous_values[layer] = (step, value)
                        for h, head in enumerate(row["heads"]):
                            for coords in grid.coordinates():
                                j = coords[0]
                                writer.writerow([name, step, layer, head, *coords,
                                    *(float(scores[m][h, j]) for m in SCORES), "" if drift is None else float(drift[h, j])])
                                report["token_rows"] += 1
                        routes = {m: select_tokens(grid, read_count, method="score", scores=scores[m].mean(dim=0).numpy()) for m in SCORES}
                        routes.update(uniform=select_tokens(grid, read_count, method="uniform"),
                                      random=select_tokens(grid, read_count, method="random", seed=42))
                        for method, route in routes.items():
                            supports.write(json.dumps(dict(input_id=name, step=step, layer=layer, method=method,
                                aggregation="mean_of_sampled_heads", heads=row["heads"], count=read_count,
                                indices=route.tolist(), frame_counts=np.bincount(route // grid.frame_size, minlength=grid.frames).tolist())) + "\n")
                            report["selections"] += 1
                            for axis, store, identity, current in (("denoising_step", previous_routes, (layer, method), step),
                                                                  ("layer", previous_layer_routes, (step, method), layer)):
                                previous = store.get(identity)
                                if previous is not None:
                                    regions = [(None, None, set(range(grid.length)))]
                                    for frame in range(grid.frames):
                                        for camera in range(grid.cameras):
                                            c = grid.coordinates()
                                            regions.append((frame, camera, set(c[(c[:, 1] == frame) & (c[:, 4] == camera), 0])))
                                    for frame, camera, region in regions:
                                        before, after = set(previous[1]) & region, set(route) & region
                                        stability.write(json.dumps(dict(input_id=name, method=method, axis=axis,
                                            fixed_layer=layer if axis == "denoising_step" else None,
                                            fixed_step=step if axis == "layer" else None,
                                            before=previous[0], after=current, adjacent=current == previous[0] + 1,
                                            frame=frame, camera=camera, before_count=len(before), after_count=len(after),
                                            jaccard=jaccard(before, after))) + "\n")
                                        report["transitions"] += 1
                                store[identity] = (current, route)
                        for method in SCORES:
                            head_routes = [select_tokens(grid, read_count, method="score", scores=scores[method][h].numpy())
                                           for h in range(len(row["heads"]))]
                            for h0, h1 in itertools.combinations(range(len(row["heads"])), 2):
                                head_stream.write(json.dumps(dict(input_id=name, step=step, layer=layer, method=method,
                                    head0=row["heads"][h0], head1=row["heads"][h1], count=read_count,
                                    jaccard=jaccard(head_routes[h0], head_routes[h1]))) + "\n")
                                report["head_comparisons"] += 1
                        report["attention_records"] += 1
        if report["validated_files"] != capture["artifacts"]:
            raise ValueError("some raw records were not independently validated")
        report.update(status="COMPLETE", exit_code=0, artifacts={name: sha256(out_dir / name)
                      for name in ("token-scores.csv", "selections.jsonl", "stability.jsonl", "head-agreement.jsonl")})
    except BaseException as exc:
        report.update(status="ERROR", exit_code=1, error=repr(exc))
        raise
    finally:
        report["end_utc"] = stamp()
        write_json(out_dir / "report.json", report)
    return report
