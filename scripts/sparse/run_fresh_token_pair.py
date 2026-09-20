#!/usr/bin/env python3
"""Bounded paired evaluation, defaulting to the historical fresh-token quick50.

Run with the frozen evaluator interpreter and its src plus LIBERO on PYTHONPATH.
Never resume/overwrite an existing output, retry a native crash, or kill foreign
processes. The existing native-read guard is enabled for every episode.
Explicit config pairs and an episode count support separately labelled pilots.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from action_eval.config import load_experiment

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def utc():
    return datetime.now(timezone.utc).isoformat()


def inventory():
    xml = ET.fromstring(subprocess.check_output(["nvidia-smi", "-q", "-x"], text=True, timeout=20))
    return [dict(index=i, uuid=g.findtext("uuid"),
        util=int(g.findtext("utilization/gpu_util").split()[0]),
        free=int(g.findtext("fb_memory_usage/free").split()[0]),
        used=int(g.findtext("fb_memory_usage/used").split()[0]),
        processes=[dict(pid=int(p.findtext("pid")), type=p.findtext("type"))
                   for p in g.findall("processes/process_info")]) for i, g in enumerate(xml.findall("gpu"))]


def admit_placement(gpus, policy_uuid, render_uuid, authorized_gpus, allow_shared_graphics=False):
    """Require the declared placement and a spare; hidden processes cannot imply an empty renderer."""
    if policy_uuid == render_uuid:
        raise ValueError("split policy and rendering GPUs")
    p = next(g for g in gpus if g["uuid"] == policy_uuid)
    r = next(g for g in gpus if g["uuid"] == render_uuid)
    if p["index"] not in authorized_gpus or r["index"] not in authorized_gpus:
        raise ValueError("placement is outside the explicitly authorized GPU indices")
    spares = [g["index"] for g in gpus if g["index"] in authorized_gpus
        and g["uuid"] not in (policy_uuid, render_uuid) and g["util"] <= 20 and g["free"] >= 35000
        and not any("G" in x["type"] for x in g["processes"])]
    empty_renderer = not r["processes"] and r["used"] <= 32
    shared_renderer = (allow_shared_graphics and bool(r["processes"])
                       and all(x["type"] == "G" for x in r["processes"]))
    ready = (p["util"] <= 20 and p["free"] >= 35000
             and not any("G" in x["type"] for x in p["processes"])
             and (empty_renderer or shared_renderer) and r["util"] <= 20
             and r["free"] >= 35000 and bool(spares))
    return spares if ready else []


def admit_cpu_rendering(gpus, policy_uuid, authorized_gpus):
    p = next(g for g in gpus if g["uuid"] == policy_uuid)
    if p["index"] not in authorized_gpus:
        raise ValueError("policy is outside the explicitly authorized GPU indices")
    spares = [g["index"] for g in gpus if g["index"] in authorized_gpus
        and g["uuid"] != policy_uuid and g["util"] <= 20 and g["free"] >= 35000
        and not any("G" in x["type"] for x in g["processes"])]
    ready = (p["util"] <= 20 and p["free"] >= 35000
             and not any("G" in x["type"] for x in p["processes"]))
    return spares if ready else []


def completed_outcomes(run_dir):
    from action_eval.runner.records import accepted_records, TASK_OUTCOME_STATUSES
    return [r for r in accepted_records(run_dir) if r.status in TASK_OUTCOME_STATUSES]


def validate_pair_configs(loaded, planned_episodes, policy_uuid, render_uuid):
    """An explicit pilot changes its manifest size, never its pairing/protocol."""
    dense, sparse = (loaded[arm] for arm in ("dense", "sparse"))
    for item in (dense, sparse):
        if item.config.benchmark.planned_episodes != planned_episodes:
            raise ValueError("config episode count differs from the declared pilot/screening size")
        runtime = item.config.runtime
        if list(runtime.policy_gpu_uuids) != [policy_uuid] or runtime.render_gpu_uuid != render_uuid:
            raise ValueError("config GPU/renderer differs from the explicit placement")
    for key in ("benchmark", "protocol", "runtime"):
        if dense.resolved[key] != sparse.resolved[key]:
            raise ValueError(f"paired configs differ in {key}")
    dpolicy, spolicy = dense.resolved["policy"], sparse.resolved["policy"]
    if {k: v for k, v in dpolicy.items() if k != "options"} != {
            k: v for k, v in spolicy.items() if k != "options"}:
        raise ValueError("paired model paths/interpreter differ")
    for key in ("action_horizon", "denoising_steps", "rng_mode", "prompt_cache"):
        if dpolicy["options"].get(key) != spolicy["options"].get(key):
            raise ValueError(f"paired common model option differs: {key}")


def fingerprint_matches(fingerprint, options, checkpoint_sha256):
    keys = ["action_horizon", "denoising_steps", "rng_mode", "prompt_cache"]
    methods = [key for key in ("visual_cache", "fresh_visual_tokens", "hybrid_visual") if key in options]
    expected = dict(options)
    if methods == ["hybrid_visual"]:
        # The adapter describes validated effective defaults, not raw YAML.
        # In particular new selector defaults and explicit reuse.mode=features
        # must compare semantically, without accepting typos or changed budgets.
        from dreamwam.sparse.hybrid import HybridConfig
        try:
            expected["hybrid_visual"] = HybridConfig.from_mapping(options["hybrid_visual"]).describe()
        except (ValueError, TypeError):
            return False
    return (len(methods) == 1 and all(fingerprint.get(k) == expected[k] for k in keys + methods)
            and fingerprint.get("checkpoint_sha256") == checkpoint_sha256)


def stop_owned_process(proc, grace_seconds=180):
    """Allow a CPU-rendered episode to finish; bound cleanup of only this process tree."""
    if proc.poll() is not None:
        return False
    os.killpg(proc.pid, signal.SIGINT)
    try:
        proc.wait(timeout=grace_seconds)
        return False
    except subprocess.TimeoutExpired:
        # Policy workers own a separate session. Snapshot descendants before
        # killing their parent; start ticks protect against PID reuse.
        snapshot = {}
        for path in Path("/proc").glob("[0-9]*/stat"):
            try:
                fields = path.read_text().rsplit(")", 1)[1].split()
                snapshot[int(path.parent.name)] = (int(fields[1]), fields[19])
            except (OSError, ValueError, IndexError):
                continue
        owned = {proc.pid}
        while True:
            expanded = owned | {pid for pid, (parent, _) in snapshot.items() if parent in owned}
            if expanded == owned:
                break
            owned = expanded
        for pid in sorted(owned - {proc.pid}, reverse=True):
            try:
                fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
                if fields[19] == snapshot[pid][1]:
                    os.kill(pid, signal.SIGKILL)
            except (OSError, IndexError):
                pass
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--adapter-report", type=Path, required=True)
    parser.add_argument("--dense-config", type=Path)
    parser.add_argument("--sparse-config", type=Path)
    parser.add_argument("--planned-episodes", type=int, default=50,
                        help="must equal each config's complete manifest; pilot configs must be labelled")
    parser.add_argument("--allow-shared-graphics", action="store_true")
    parser.add_argument("--render-backend", choices=("egl", "osmesa"), default="egl")
    parser.add_argument("--first-arm", choices=("sparse", "dense"), default="sparse",
                        help="use Dense first when establishing a new renderer baseline")
    parser.add_argument("--authorized-gpus", type=int, nargs="+", default=[4, 5, 6, 7],
                        help="explicitly authorized physical indices for this host, including an unused spare")
    parser.add_argument("--wall-seconds", type=int, default=3600)
    parser.add_argument("--admission-seconds", type=int, default=120)
    parser.add_argument("--stop-grace-seconds", type=int, default=180)
    parser.add_argument("--smoke-only", action="store_true",
                        help="stop after the first accepted candidate episode; retain its 50-identity manifest")
    args = parser.parse_args()
    if bool(args.dense_config) != bool(args.sparse_config):
        parser.error("supply both explicit configs or neither")
    if args.planned_episodes < 1 or (not args.dense_config and args.planned_episodes != 50):
        parser.error("non-50 pilots require explicit labelled configs and a positive episode count")
    if len(set(args.authorized_gpus)) < 3 or any(i < 0 for i in args.authorized_gpus):
        parser.error("authorize at least three distinct nonnegative GPU indices")
    if min(args.wall_seconds, args.admission_seconds, args.stop_grace_seconds) <= 0:
        parser.error("time limits must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    report = json.loads(args.adapter_report.read_text())
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.model_root, text=True).strip()
    if report["status"] != "VERIFIED" or report["commit"]["stdout"] != commit:
        raise ValueError("require verified adapter report for this frozen model")
    meta = dict(status="ADMISSION", start_utc=utc(), pid=os.getpid(), model_commit=commit,
        evaluator_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.eval_root, text=True).strip(),
        planned_episodes_per_arm=args.planned_episodes, shared_graphics=args.allow_shared_graphics,
        authorized_gpu_indices=args.authorized_gpus,
        render_backend=args.render_backend,
        first_arm=args.first_arm,
        wall_seconds_per_arm=args.wall_seconds, stop_grace_seconds=args.stop_grace_seconds,
        smoke_only=args.smoke_only, runs=[], sr=None)
    save = lambda: (args.out_dir / "controller.json").write_text(json.dumps(meta, indent=2) + "\n")
    env = os.environ.copy()
    env.update(MODEL_ROOT=str(args.model_root), MUJOCO_GL=args.render_backend,
               PYOPENGL_PLATFORM=args.render_backend, OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    env.pop("MUJOCO_EGL_DEVICE_ID", None)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    os.environ["MODEL_ROOT"] = str(args.model_root)
    # GPU_UUID, RENDER_GPU_UUID and LIBERO_ROOT remain explicit caller inputs.
    policy_uuid = env["GPU_UUID"]
    render_uuid = env["RENDER_GPU_UUID"] if args.render_backend == "egl" else None
    if policy_uuid == render_uuid:
        raise ValueError("split policy and rendering GPUs")
    suffix = "-osmesa" if args.render_backend == "osmesa" else ""
    configs = dict(dense=args.dense_config or args.eval_root / f"configs/experiments/dreamwam-fresh-control-quick50{suffix}.yaml",
                   sparse=args.sparse_config or args.eval_root / f"configs/experiments/dreamwam-fresh-token10-quick50{suffix}.yaml")
    loaded_configs = {arm: load_experiment(path, require_paths=True) for arm, path in configs.items()}
    validate_pair_configs(loaded_configs, args.planned_episodes, policy_uuid, render_uuid)
    for arm, loaded in loaded_configs.items():
        reference = report["arms"][arm]["fingerprint"]
        if not fingerprint_matches(reference, loaded.config.policy.options, reference["checkpoint_sha256"]):
            raise ValueError("adapter report does not validate the requested model options")
    meta["configs"] = {arm: str(path) for arm, path in configs.items()}

    def record(gpus, arm, phase):
        row = dict(utc=utc(), arm=arm, phase=phase, gpus=gpus)
        with (args.out_dir / "gpus.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        return row

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, interrupt)
    save()
    try:
        arms = [("sparse", "token10"), ("dense", "control")]
        if args.first_arm == "dense":
            arms.reverse()
        for arm, kind in arms:
            config, loaded = configs[arm], loaded_configs[arm]
            deadline = time.monotonic() + args.admission_seconds
            while True:
                gpus = inventory()
                row = record(gpus, arm, "admission")
                r = next((g for g in gpus if g["uuid"] == render_uuid), None)
                spares = (admit_cpu_rendering(gpus, policy_uuid, args.authorized_gpus)
                          if args.render_backend == "osmesa" else
                          admit_placement(gpus, policy_uuid, render_uuid,
                                          args.authorized_gpus, args.allow_shared_graphics))
                if spares:
                    row["spares_unused"] = spares
                    break
                if time.monotonic() >= deadline:
                    meta.update(status="RESOURCE_WINDOW_CLOSED", waiting_arm=arm)
                    return
                time.sleep(3)
            allowed_graphics = {x["pid"] for x in r["processes"]} if r else set()
            root = args.out_dir / arm
            root.mkdir()
            rendering = ["--render-cpu"] if render_uuid is None else ["--render-gpu", render_uuid]
            cmd = [str(args.eval_root / ".venv/bin/python"), "scripts/supplement-libero.py", str(config),
                   *rendering, "--output", str(root), "--check-native-writes"]
            entry = dict(arm=arm, start_utc=utc(), command=cmd, admission=row,
                         output=str(root), fingerprint_verified=False)
            meta["runs"].append(entry)
            with (root / "launch.log").open("w") as log:
                proc = subprocess.Popen(cmd, cwd=args.eval_root, env=env, stdout=log,
                                        stderr=subprocess.STDOUT, start_new_session=True)
                entry["pid"] = proc.pid
                meta["status"] = "RUNNING"
                save()
                deadline = time.monotonic() + args.wall_seconds
                violation = None
                try:
                    while proc.poll() is None:
                        gpus = inventory()
                        record(gpus, arm, "running")
                        renderer = next((g for g in gpus if g["uuid"] == render_uuid), None)
                        if renderer and any("C" in x["type"] or x["pid"] not in allowed_graphics | {proc.pid} for x in renderer["processes"]):
                            violation = "renderer_resource_window_closed"
                            break
                        if time.monotonic() >= deadline:
                            violation = "wall_limit"
                            break
                        path = root / "run/provenance.json"
                        if not entry["fingerprint_verified"] and path.exists():
                            fingerprint = json.loads(path.read_text()).get("policy_worker", {}).get("description", {}).get("fingerprint")
                            if fingerprint:
                                expected = loaded.config.policy.options
                                if not fingerprint_matches(fingerprint, expected, report["arms"][arm]["fingerprint"]["checkpoint_sha256"]):
                                    violation = "policy_fingerprint_mismatch"
                                    break
                                entry["fingerprint_verified"] = True
                                save()
                        if args.smoke_only and completed_outcomes(root / "run"):
                            violation = "planned_smoke_stop_after_first_accepted_episode"
                            break
                        time.sleep(3)
                finally:
                    entry["forced_cleanup"] = stop_owned_process(proc, args.stop_grace_seconds)
                    entry.update(exit_code=proc.returncode, end_utc=utc(), stop_reason=violation)
                    (root / "launch.exit").write_text(str(proc.returncode) + "\n")
                    save()
            if violation or proc.returncode != 0 or not entry["fingerprint_verified"]:
                meta["status"] = "SMOKE_STOPPED_FOR_REVIEW" if args.smoke_only else "STOPPED_REQUIRES_REVIEW"
                return
            manifest = json.loads((root / "run/manifest.json").read_text())
            accepted = completed_outcomes(root / "run")
            if manifest["planned_episodes"] != args.planned_episodes or len(accepted) != args.planned_episodes:
                raise ValueError("run exited without all declared accepted outcomes")
        meta["status"] = "QUICK50_PAIR_COMPLETE" if args.planned_episodes == 50 else "PILOT_PAIR_COMPLETE"
    except KeyboardInterrupt:
        meta["status"] = "INTERRUPTED"
        raise
    except BaseException as exc:
        meta.update(status="ERROR", error=repr(exc))
        raise
    finally:
        meta["end_utc"] = utc()
        save()


if __name__ == "__main__":
    main()
