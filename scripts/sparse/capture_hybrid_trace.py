#!/usr/bin/env python3
"""Real-checkpoint, complete-history adapter replay and bitwise-checked trace gallery."""

import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from observation_sequences import load_sequences, sha256
from trace_gallery import render_capture
from dreamwam.policy import _center_crop_resize
from dreamwam.sparse.hybrid.trace import HybridTrace
from evaluation.action_eval.infer import create_policy


@contextmanager
def input_capture(policy, observation):
    arrays = {"raw_" + key: value.copy() for key, value in observation["images"].items()}
    arrays["state"] = observation["state"].copy()
    arrays["model_input_rgb"] = np.concatenate([
        _center_crop_resize(observation["images"][name], policy.image_size)
        for name in ("agentview", "wrist")], axis=1)
    native = policy.vae.single_encode
    def capture(tensor, *args, **kwargs):
        if "vae_input" in arrays:
            raise AssertionError("multiple first-frame encodes in a chunk")
        expected = torch.from_numpy(arrays["model_input_rgb"].copy()).permute(2, 0, 1).unsqueeze(0)
        expected = expected.to(device=tensor.device, dtype=tensor.dtype).mul(2.0 / 255.0).sub(1.0).unsqueeze(2)
        if not torch.equal(tensor, expected):
            raise AssertionError("plotted pixels do not align with the actual VAE input")
        arrays["vae_input"] = tensor.float().cpu().numpy().copy()
        return native(tensor, *args, **kwargs)
    policy.vae.single_encode = capture
    try:
        yield arrays
        if "vae_input" not in arrays:
            raise AssertionError("first-frame VAE input was not captured")
    finally:
        policy.vae.single_encode = native


def _decision(stats):
    # Wall-clock cost and monotonically increasing request identity are not decisions.
    result = deepcopy(stats)
    result.pop("controller_seconds", None)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--options", type=Path, default=Path("configs/sparse/m1-fixed-m2-m3.json"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--layers", default="0,29")
    parser.add_argument("--steps", default="0,5,9")
    parser.add_argument("--head", type=int, default=0)
    parser.add_argument("--query", type=int, default=0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("real-checkpoint capture requires CUDA; refusing CPU fallback")
    root = Path(__file__).resolve().parents[2]
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip():
        raise RuntimeError("capture requires a clean committed worktree")
    manifest, sequences = load_sequences(args.inputs)
    original_options = json.loads(args.options.read_text())
    options = deepcopy(original_options)
    options["hybrid_visual"]["execution"]["backend"] = "eager"
    options["hybrid_visual"]["diagnostics"] = {"level": "trace"}
    layers = tuple(map(int, args.layers.split(",")))
    steps = tuple(map(int, args.steps.split(",")))
    args.out_dir.mkdir(parents=True, exist_ok=False)
    report = dict(status="RUNNING", start_utc=datetime.now(timezone.utc).isoformat(), argv=sys.argv,
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        source_options=original_options, effective_capture_options=options,
        options_sha256=sha256(args.options), inputs_sha256=sha256(args.inputs),
        source_episode_split_sha256=manifest["episode_split_sha256"], split=manifest["split_role"],
        cohort="H200 model replay of captured H100-rendered development observations",
        torch=torch.__version__, cuda=torch.version.cuda, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        gpu_name=torch.cuda.get_device_name(), episodes=[], sr=None, latency_claim=None,
        limitations=["offline observed-input replay; not candidate closed-loop states or success",
            "instrumentation reconstructs probabilities from actual Q/K; fused internal probabilities unavailable",
            "all-head/query arrays retained; displayed examples do not establish causal token importance"])
    write = lambda: (args.out_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    adapter = None
    try:
        write()
        adapter = create_policy(dict(model_root=str(root), model_config=str(root / "configs/dreamwam_joint.yaml"),
            checkpoint=str(root / "checkpoints/dreamwam_joint.pt"), hash_checkpoint=True, **options), "cuda")
        report["fingerprint"] = adapter.describe().fingerprint
        checkpoint_hash = report["fingerprint"]["checkpoint_sha256"]
        if checkpoint_hash != "6c087e5b9e201f19dbe92834b470a7ee18937a18ea9adcbb49886aa1e3ac4c61":
            raise AssertionError("checkpoint differs from the frozen reference")
        model = adapter.policy.model
        versions = {name: parameter._version for name, parameter in model.named_parameters()}
        evaluation = dict(adapter.policy.evaluation)
        runtime = adapter.policy._hybrid_visual_runtime
        if not 0 <= args.head < model.mot.num_heads or not 0 <= args.query < adapter._action_horizon:
            raise ValueError("head or action query is outside the real model")
        for identity, sequence in sequences.items():
            episode_id = "t%03d-i%03d-r%02d-s%d" % identity
            reset = dict(episode_id=episode_id, seed=identity[-1])
            adapter.reset(reset)
            refs = []
            for entry, observation in sequence:
                prediction = adapter.predict(observation)
                refs.append((prediction.actions.copy(), deepcopy(prediction.diagnostics)))
            adapter.reset(reset)
            chosen = {0, len(sequence) // 2, len(sequence) - 1}
            rows = []
            for index, ((entry, observation), (reference, reference_stats)) in enumerate(zip(sequence, refs)):
                directory = args.out_dir / f"{episode_id}-c{entry['call_index']:04d}"
                if index in chosen:
                    with input_capture(adapter.policy, observation) as inputs, HybridTrace(runtime, layers=layers, steps=steps) as trace:
                        prediction = adapter.predict(observation)
                else:
                    prediction = adapter.predict(observation)
                if not np.array_equal(prediction.actions, reference):
                    raise AssertionError("instrumentation changed actions at " + entry["id"])
                stats = prediction.diagnostics
                if "chunk_budget" in stats and _decision(stats["chunk_budget"]) != _decision(reference_stats["chunk_budget"]):
                    raise AssertionError("instrumentation changed the causal budget decision")
                for before, after in zip(reference_stats["hybrid_visual"]["steps"], stats["hybrid_visual"]["steps"]):
                    for key in ("effective_op", "q_rows", "kv_rows", "route", "query"):
                        if before[key] != after[key]:
                            raise AssertionError("instrumentation changed executed work")
                if runtime.state.output is not None or runtime._active:
                    raise AssertionError("visual cache leaked past the chunk boundary")
                row = dict(input_id=entry["id"], input_sha256=entry["sha256"], call_index=entry["call_index"],
                    environment_step=entry["step_index"], bitwise_uninstrumented=True,
                    chunk_budget=stats.get("chunk_budget"), hybrid_visual=stats["hybrid_visual"])
                if index in chosen:
                    directory.mkdir()
                    np.savez_compressed(directory / "tensors.npz", **inputs, **trace.arrays,
                                        actions=prediction.actions, uninstrumented_actions=reference)
                    budget = stats["hybrid_visual"]
                    metadata = dict(episode_id=episode_id, call_index=entry["call_index"], input=entry,
                        instruction=observation["instruction"], grid=trace.grid, records=trace.records,
                        commit=report["commit"], checkpoint_sha256=checkpoint_hash,
                        budget_label=f"{budget['query_ratio']:.0%}/{budget['read_ratio']:.0%}",
                        diagnostics=stats, actual_vae_input_verified=True, bitwise_uninstrumented=True,
                        tensors_sha256=sha256(directory / "tensors.npz"))
                    (directory / "trace.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
                    row["capture"] = directory.name
                rows.append(row)
                print(json.dumps(dict(episode=episode_id, call=entry["call_index"], capture=index in chosen,
                                      level=stats.get("chunk_budget", {}).get("level"), bitwise=True)), flush=True)
            report["episodes"].append(dict(episode_id=episode_id, calls=rows))
            write()
        if evaluation != adapter.policy.evaluation or versions != {n: p._version for n, p in model.named_parameters()}:
            raise AssertionError("evaluation settings or model weights changed")
        adapter.close(); adapter = None
        for episode in report["episodes"]:
            for row in episode["calls"]:
                if "capture" in row:
                    row["artifacts"] = render_capture(args.out_dir / row["capture"], head=args.head, query=args.query)
        links = "\n".join(f'<li><a href="{row["capture"]}/index.html">{html.escape(row["capture"])}</a></li>'
            for episode in report["episodes"] for row in episode["calls"] if "capture" in row)
        (args.out_dir / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>M1/M2/M3 evidence</title>'
            '<h1>Real execution traces</h1><p>Complete observation histories; instrumented eager capture, bitwise-checked actions. '
            'No SR or speed claim. <a href="report.json">Provenance and decisions</a>.</p><ul>' + links + '</ul>')
        report.update(status="VERIFIED_REAL_CHECKPOINT_TRACE", exit_code=0)
    except BaseException as exc:
        report.update(status="ERROR", error=repr(exc), exit_code=1)
        raise
    finally:
        if adapter is not None: adapter.close()
        report["end_utc"] = datetime.now(timezone.utc).isoformat()
        write()
    print(json.dumps(dict(status=report["status"], episodes=len(report["episodes"]), sr=None)))


if __name__ == "__main__":
    main()
