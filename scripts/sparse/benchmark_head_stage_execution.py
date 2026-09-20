#!/usr/bin/env python3
"""One factor: compact SDPA versus the same frozen Head/Stage mask budgets.

All complete requests retain the released model, ten steps and original RNG.
Graph and eager measurements are separate runs. Native and compact full-budget
controls expose splitting overhead/numerics; an existing conditioned-frame
Dense control also prevents a weak-Dense headline. No SR is inferred.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

from benchmark_ffn_context_cache import command, sha256, summary
from collect_action_impact import components
from compare_head_stage_allocations import allocations
from dreamwam.config import load_release_config
from dreamwam.policy import build_policy
from dreamwam.sparse.conditioned_frame_cache import ConditionedFrameCache, GraphedConditionedFrameCache
from dreamwam.sparse.head_stage_execution import HeadStageExecution, HeadStageGraph


def relative_l2(actual, reference):
    actual, reference = np.asarray(actual, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    return float(np.linalg.norm(actual - reference) / max(np.linalg.norm(reference), 1e-12))


class FinalOutputs:
    """Observe final outputs only in untimed checks; no attention intervention."""
    def __init__(self, model):
        self.model = model
        self.raw = self.video = None
        self.steps = 0

    def __enter__(self):
        self.sample, self.step = self.model.sample_action, self.model.video_scheduler.step

        def sample(*args, **kwargs):
            output = self.sample(*args, **kwargs)
            self.raw = output.detach().clone()
            return output

        def step(*args, **kwargs):
            self.video = self.step(*args, **kwargs)
            self.steps += 1
            return self.video

        self.model.sample_action, self.model.video_scheduler.step = sample, step
        return self

    def __exit__(self, *args):
        self.model.sample_action, self.model.video_scheduler.step = self.sample, self.step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dreamwam_joint.yaml")
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--classification", type=Path, required=True)
    parser.add_argument("--mode", choices=("eager", "graph"), required=True)
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.reps < 30 or args.reps % 30 or args.warmup < 1:
        parser.error("use positive multiples of 30 for ten positions and three inputs")
    dataset, classes = (json.loads(path.read_text()) for path in (args.inputs, args.classification))
    ids = [r["id"] for r in dataset["inputs"]]
    if (dataset["status"] != "CAPTURED" or len(ids) != 3 or
            classes["status"] != "EXPLORATORY_CLASSIFICATION" or
            classes["calibration_inputs"] != ids[:2] or classes["check_input"] != ids[2] or
            classes["input_manifest_sha256"] != sha256(args.inputs)):
        raise ValueError("require the frozen two-input calibration and separate check input")
    release = load_release_config(args.config)
    if sha256(release.paths.checkpoint) != classes["checkpoint_sha256"]:
        raise ValueError("checkpoint changed")
    profiles = {k: v for k, v in allocations(classes["entries"]).items()
                if k in {"dense", "uniform", "head_only", "head_stage"}}
    args.out_dir.mkdir(parents=True, exist_ok=False)
    inventory = lambda: command("nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,memory.used", "--format=csv")
    manifest = dict(status="RUNNING", start_utc=datetime.now(timezone.utc).isoformat(),
        argv=sys.argv, pid=os.getpid(), git=command("git", "rev-parse", "HEAD"),
        dirty=command("git", "status", "--porcelain"), python=platform.python_version(),
        torch=torch.__version__, cuda=torch.version.cuda, mode=args.mode,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), gpu_before=inventory(),
        checkpoint_sha256=classes["checkpoint_sha256"], input_manifest_sha256=sha256(args.inputs),
        classification_sha256=sha256(args.classification), profiles=profiles,
        sources={str(p): sha256(p) for p in [__file__, args.config, "dreamwam/mot.py",
            "dreamwam/sparse/head_stage_execution.py", "dreamwam/sparse/conditioned_frame_cache.py",
            "dreamwam/sparse/visual_cache_graphs.py", "scripts/sparse/compare_head_stage_allocations.py"]},
        factor="compact per-budget SDPA versus identical masked future-key selections",
        latency_boundary="synchronized full predict_action through CPU actions; no output observers in timing",
        environment="existing environment reused without installs, sync or dependency changes",
        calibration_inputs=ids[:2], check_input=ids[2], sr=None,
        limitations=["Three exposed initial scenes; action/video proxies are not SR",
            "One fixed 50% future-key budget; no AV selection or temporal reuse",
            "Matrix extents are submitted SDPA pairs, not hardware FLOPs",
            "Conditioned Dense is an additional stronger reference; backend comparisons use the same budget",
            "Numerical splitting differences must be disclosed; no backend equivalence inferred from full-budget parity"],
        cold_requests=[], quality=[], timed_requests=0)
    write = lambda: (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write()
    graphs = {}
    try:
        policy = build_policy(release, device="cuda")
        evaluation = dict(policy.evaluation)
        versions = {name: p._version for name, p in policy.model.named_parameters()}
        if evaluation["denoising_steps"] != 10 or evaluation["action_horizon"] != 32:
            raise ValueError("retain the scientific sampler protocol")
        manifest.update(evaluation=evaluation, dtype=str(policy.dtype))
        inputs = []
        for entry in dataset["inputs"]:
            path = args.inputs.parent / entry["path"]
            if sha256(path) != entry["sha256"]:
                raise ValueError("captured input changed")
            with np.load(path, allow_pickle=False) as data:
                inputs.append(dict(images={k: data[k].copy() for k in ["agentview", "wrist"]},
                    state=data["state"].copy(), instruction=str(data["instruction"].item())))
        controllers = {f"{backend}_{name}": HeadStageExecution(policy.model.mot, profile, backend=backend)
                       for name, profile in profiles.items() for backend in ("masked", "compact")}
        variants = ("native_dense", *controllers, "conditioned_dense")
        manifest.update(variants=variants, reps_per_variant=args.reps)
        if args.mode == "graph":
            graphs = {name: HeadStageGraph(policy.model) for name in variants if name != "conditioned_dense"}
            graphs["conditioned_dense"] = GraphedConditionedFrameCache(policy.model, refresh_every=1)
        conditioned_eager = ConditionedFrameCache(policy.model, refresh_every=1)

        def predict(name, input_id, *, graph=False, observe=False):
            cache = graphs[name] if graph else conditioned_eager if name == "conditioned_dense" else nullcontext()
            with controllers.get(name, nullcontext()), cache as replay:
                observer = FinalOutputs(policy.model) if observe else nullcontext()
                with observer as captured:
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    action = policy.predict_action(**inputs[input_id])
                    torch.cuda.synchronize()
                    seconds = time.perf_counter() - started
                    stats = dict(replay.last_stats) if replay is not None else {}
                    extra = None
                    if observe:
                        if captured.steps != 10 or captured.raw is None or captured.video is None:
                            raise AssertionError("incomplete ten-step output capture")
                        extra = dict(raw=captured.raw.float().cpu().numpy(), video=captured.video.float().cpu().numpy())
                        if not all(np.isfinite(v).all() for v in extra.values()):
                            raise FloatingPointError("non-finite raw output")
            if not np.isfinite(action).all():
                raise FloatingPointError(name)
            if graph and name != "conditioned_dense" and stats != dict(stage_calls=[3, 4, 3], graph_replays=10):
                raise AssertionError("missing stage-specific graph replay")
            return seconds, action, extra, stats

        references = {}
        # All candidates get their own eager reference on the same real bytes.
        for input_id in range(3):
            for name in variants:
                _, action, extra, _ = predict(name, input_id, observe=True)
                references[name, input_id] = dict(action=action.copy(), **extra)
                native = references["native_dense", input_id]
                masked_name = name.replace("compact_", "masked_")
                masked = references.get((masked_name, input_id), native)
                row = dict(input_id=ids[input_id], variant=name,
                    normalized_action_relative_l2=relative_l2(extra["raw"], native["raw"]),
                    future_video_relative_l2=relative_l2(extra["video"][:, :, 1:], native["video"][:, :, 1:]),
                    action=components(native["action"], action),
                    prefix10_action_relative_l2=relative_l2(action[:10], native["action"][:10]),
                    masked_action_relative_l2=relative_l2(extra["raw"], masked["raw"]),
                    masked_video_relative_l2=relative_l2(extra["video"][:, :, 1:], masked["video"][:, :, 1:]),
                    conditioned_frame_bitwise=bool(np.array_equal(extra["video"][:, :, :1], native["video"][:, :, :1])),
                    actions=action.tolist(), normalized_actions=extra["raw"].tolist())
                if not row["conditioned_frame_bitwise"]:
                    raise AssertionError("sampler conditioned latent frame changed")
                if name == "masked_dense" and (not np.array_equal(action, native["action"]) or
                                                not np.array_equal(extra["video"], native["video"])):
                    raise AssertionError("full masked allocation lost native parity")
                manifest["quality"].append(row)
                write()
        # Revisit input zero after different images/instructions and graph captures.
        for request, input_id in enumerate((0, 1, 2, 0)):
            for name in variants:
                seconds, action, extra, stats = predict(name, input_id, graph=args.mode == "graph", observe=True)
                reference = references[name, input_id]
                if not (np.array_equal(action, reference["action"]) and
                        np.array_equal(extra["raw"], reference["raw"]) and
                        np.array_equal(extra["video"], reference["video"])):
                    raise AssertionError(f"own-eager output/request isolation failed: {name}, input {input_id}")
                if request == 0:
                    manifest["cold_requests"].append(dict(variant=name, seconds=seconds, stats=stats))
                write()
            print(f"input {input_id}: all ten variants passed own-eager output parity", flush=True)
        for name in variants:
            for rep in range(args.warmup):
                predict(name, rep % 3, graph=args.mode == "graph")
        samples = {name: [] for name in variants}
        with (args.out_dir / "requests.jsonl").open("w", buffering=1) as journal:
            for rep in range(args.reps):
                order = variants[rep % len(variants):] + variants[:rep % len(variants)]
                for name in order:
                    seconds, action, _, stats = predict(name, rep % 3, graph=args.mode == "graph")
                    if not np.array_equal(action, references[name, rep % 3]["action"]):
                        raise AssertionError(f"timed output parity failed: {name}")
                    samples[name].append(seconds)
                    journal.write(json.dumps(dict(repeat=rep, order=order, variant=name,
                        input_id=ids[rep % 3], seconds=seconds, stats=stats, bitwise_own_eager=True)) + "\n")
                    manifest["timed_requests"] += 1
                write()
                print(f"timed round {rep + 1}/{args.reps}", flush=True)
        if evaluation != policy.evaluation or versions != {n: p._version for n, p in policy.model.named_parameters()}:
            raise AssertionError("weights/evaluation changed")
        metrics = {name: summary(values) for name, values in samples.items()}
        means = {name: metric["mean"] for name, metric in metrics.items()}
        report = dict(status="MEASURED", end_utc=datetime.now(timezone.utc).isoformat(),
            mode=args.mode, seconds=metrics,
            compact_vs_identical_mask={name: means[f"masked_{name}"] / means[f"compact_{name}"] for name in profiles},
            vs_native_dense={name: means["native_dense"] / value for name, value in means.items()},
            vs_conditioned_dense={name: means["conditioned_dense"] / value for name, value in means.items()},
            graphs={name: cache.graph_stats() for name, cache in graphs.items()},
            sdpa_matrix_pairs={name: sorted({p.matrix_pairs if c.backend == "compact" else
                len(p.budgets) * p.joint_length ** 2 for p in c.plans.values()}) for name, c in controllers.items()},
            all_timed_actions_bitwise_own_eager=True, peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
            peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20, gpu_after=inventory(), sr=None)
        (args.out_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        manifest.update(status="MEASURED", end_utc=report["end_utc"])
        write()
        print(json.dumps(report, indent=2), flush=True)
    except BaseException as exc:
        manifest.update(status="ERROR", error=repr(exc), end_utc=datetime.now(timezone.utc).isoformat())
        write()
        raise
    finally:
        for cache in graphs.values():
            cache.close_graphs()


if __name__ == "__main__":
    main()
