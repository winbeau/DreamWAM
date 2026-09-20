"""Frozen, bounded counterfactual design and raw-output diagnostics."""

import json
from pathlib import Path

import numpy as np

from .analysis import index_capture, load_record
from .archive import sha256
from .geometry import TokenGrid
from .selection import select_tokens


def case_specs(design):
    names = design["input_ids"]
    if len(names) != 2 or len(set(names)) != 2:
        raise ValueError("bounded design requires exactly two distinct development observations")
    primary, secondary = names
    step, layer = design["primary_step_layer"]
    cases = []

    def add(name, methods, scopes, operation="delete", location=(step, layer), bottom=False):
        for method in methods:
            for scope in scopes:
                cases.append(dict(input_id=name, method=method, scope=scope, operation=operation,
                                  step=location[0], layer=location[1], bottom=bottom))

    add(primary, design["primary_methods"], ("AV", "VV", "joint"))
    add(secondary, design["secondary_methods"], ("AV", "VV"))
    for location in design["endpoint_step_layers"]:
        add(primary, ("uniform", "value_video_time"), ("AV", "VV"), location=location)
    for operation in ("replace_value_zero", "recompute"):
        add(primary, ("uniform", "value_action", "value_video_time"), ("AV",), operation=operation)
    add(primary, ("value_video_time",), ("AV", "VV"), bottom=True)
    add(primary, ("value_norm",), ("AV", "VV"))
    if len({json.dumps(row, sort_keys=True) for row in cases}) != len(cases):
        raise ValueError("duplicate diagnostic cells")
    if len(cases) + len(names) > min(design["max_predict_calls"], 49):
        raise ValueError("bounded diagnostic call cap exceeded; no automatic expansion")
    return cases


def prepare_cases(plan, capture_root, analysis_root):
    capture_root, analysis_root = Path(capture_root), Path(analysis_root)
    analysis = json.loads((analysis_root / "report.json").read_text())
    if (analysis.get("status") != "COMPLETE" or analysis.get("exit_code") != 0 or
        analysis.get("input_kind") != "self_captured_observations" or analysis.get("split_role") != "development"):
        raise ValueError("diagnostics require verified self-captured development analysis")
    if sha256(capture_root / "report.json") != analysis["capture_report_sha256"]:
        raise ValueError("capture differs from verified analysis")
    capture, indexed = index_capture(capture_root)
    if capture["checkpoint_sha256"] != plan["checkpoint_sha256"]:
        raise ValueError("diagnostic checkpoint differs from the capture")
    routes_path = analysis_root / "selections.jsonl"
    if sha256(routes_path) != analysis["artifacts"]["selections.jsonl"]:
        raise ValueError("frozen routes differ from raw replay")
    design = plan["diagnostic_study"]
    if analysis["read_count"] != design["selected_count"]:
        raise ValueError("diagnostic count differs from the profiled equal budget")
    routes = {}
    for line in routes_path.read_text().splitlines():
        row = json.loads(line)
        identity = (row["input_id"], row["step"], row["layer"], row["method"])
        if identity in routes:
            raise ValueError("duplicate frozen route")
        routes[identity] = row
    entries = {entry["input_id"]: entry for entry in capture["inputs"]}
    cases = case_specs(design)
    for i, case in enumerate(cases):
        grid = TokenGrid(*entries[case["input_id"]]["grid"])
        key = (case["input_id"], case["step"], case["layer"], case["method"])
        if case["bottom"]:
            record = indexed[case["input_id"], "attention", case["step"], case["layer"]]
            raw = load_record(capture_root / "raw", record)
            indices = select_tokens(grid, design["selected_count"], method="score", bottom=True,
                scores=raw["score_" + case["method"]].mean(axis=0)).tolist()
        else:
            indices = routes[key]["indices"]
        if len(set(indices)) != design["selected_count"] or any(type(j) is not int or not 0 <= j < grid.length for j in indices):
            raise ValueError("route violates the frozen original-position budget")
        counts = np.bincount(np.array(indices) // grid.frame_size, minlength=grid.frames).tolist()
        expected = [design["selected_count"] // grid.frames + int(f < design["selected_count"] % grid.frames)
                    for f in range(grid.frames)]
        if counts != expected:
            raise ValueError("diagnostic route changes the balanced frame budget")
        case.update(case_id=f"case-{i:03d}", indices=indices, count=len(indices), frame_counts=counts,
            route_source="offline_native_dense_profile; never an online teacher input",
            count_semantics="current K/V rows, all others use previous-step K/V" if case["operation"] == "recompute"
                            else "perturbed original visual-key rows")
    return dict(cases=cases, planned_calls=len(cases) + len(design["input_ids"]),
        checkpoint_sha256=capture["checkpoint_sha256"], capture_root=str(capture_root.resolve()),
        analysis_root=str(analysis_root.resolve()), capture_report_sha256=sha256(capture_root / "report.json"),
        analysis_report_sha256=sha256(analysis_root / "report.json"), raw_index_sha256=capture["raw_index_sha256"],
        routes_sha256=sha256(routes_path), input_manifest_sha256=capture["input_manifest_sha256"],
        input_ids=design["input_ids"], input_kind="self_captured_observations", split_role="development")


def error_metrics(actual, reference):
    actual, reference = np.asarray(actual, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    if actual.shape != reference.shape or not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise ValueError("diagnostics require matching finite raw outputs")
    delta = actual - reference
    return dict(relative_l2=float(np.linalg.norm(delta) / max(np.linalg.norm(reference), 1e-12)),
                rmse=float(np.sqrt(np.mean(delta**2))), maximum_absolute=float(np.max(np.abs(delta))),
                bitwise_equal=bool(np.array_equal(actual, reference)))


def output_diagnostics(actual, reference, *, prefix=10):
    if actual["action"].shape[0] < prefix or actual["video_latents"].shape[2] < 2:
        raise ValueError("diagnostic output omits executed prefix or future video")
    result = dict(action=error_metrics(actual["action"], reference["action"]),
        executed_prefix=error_metrics(actual["action"][:prefix], reference["action"][:prefix]),
        raw_action=error_metrics(actual["raw_action"], reference["raw_action"]),
        raw_prefix=error_metrics(actual["raw_action"][:, :prefix], reference["raw_action"][:, :prefix]),
        future_latents=error_metrics(actual["video_latents"][:, :, 1:], reference["video_latents"][:, :, 1:]))
    for label, length in (("action", actual["action"].shape[0]), ("executed_prefix", prefix)):
        result[label]["gripper_sign_disagreements"] = int(np.count_nonzero(
            np.sign(actual["action"][:length, -1]) != np.sign(reference["action"][:length, -1])))
    return result
