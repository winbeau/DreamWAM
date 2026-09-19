#!/usr/bin/env python3
"""Launch an action-eval run and refuse to let it proceed unless it runs what it claims.

This automates the manual sequence that caught two real incidents on this project:

1. an experiment YAML declared ``sparse.enabled: true`` and the platform passed the options
   through, but the model checkout predated the adapter change that consumes them, so the
   run executed dense while being labelled sparse;
2. a run was started on a card another user's job had just taken, where the measured
   slowdown risks tripping ``predict_timeout_s`` and recording winnable episodes as errors.

Both are silent by default. This launcher makes them loud, and stops the run before it
consumes GPU hours:

* refuses to start unless the target GPU is idle (shared-card timings are worthless and
  timeouts corrupt SR);
* starts the queue detached, then waits for the worker's ``describe()`` fingerprint;
* asserts the *executed* sparse configuration and denoising-step count equal what the
  experiment YAML declares, and kills the run if they do not.

Run it on the evaluation server, under the model checkout's environment.

    .venv/bin/python scripts/sparse/launch_run.py \
        --lane sparse-gpu7 --gpu 7 \
        --config-stem dreamwam-sparse-av25-libero-spatial
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--config-stem", action="append", required=True)
    parser.add_argument(
        "--max-foreign-mib",
        type=int,
        default=0,
        help=(
            "refuse to start if any other process holds more than this much memory on the "
            "card; 0 means the card must be idle"
        ),
    )
    parser.add_argument("--verify-timeout", type=float, default=1200.0)
    parser.add_argument("--ae-root", default=os.environ.get("AE_ROOT"))
    parser.add_argument("--model-root", default=os.environ.get("MODEL_ROOT"))
    parser.add_argument("--libero-root", default=os.environ.get("LIBERO_ROOT"))
    parser.add_argument("--out-root", default=os.environ.get("OUT_ROOT"))
    parser.add_argument("--log-dir", default=None)
    return parser.parse_args()


def gpu_foreign_mib(gpu: int) -> tuple[int, list[str]]:
    """Total memory held by other processes on ``gpu``, plus their descriptions."""
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                str(gpu),
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except subprocess.CalledProcessError as error:
        raise SystemExit(f"nvidia-smi failed for GPU {gpu}: {error.stderr.strip()}")
    total = 0
    descriptions: list[str] = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [piece.strip() for piece in line.split(",")]
        if len(parts) < 3:
            continue
        pid, name = parts[0], parts[1]
        try:
            total += int(parts[2])
        except ValueError:
            continue
        descriptions.append(f"{pid} {name}")
    return total, descriptions


def read_experiment(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text())
    options = (payload.get("policy") or {}).get("options") or {}
    return {
        "sparse": options.get("sparse"),
        "denoising_steps": options.get("denoising_steps"),
        "action_horizon": options.get("action_horizon"),
    }


def worker_fingerprint(run_dir: Path) -> dict | None:
    provenance = run_dir / "provenance.json"
    if not provenance.is_file():
        return None
    try:
        payload = json.loads(provenance.read_text())
    except json.JSONDecodeError:
        return None
    worker = payload.get("policy_worker") or {}
    description = worker.get("description") or {}
    fingerprint = description.get("fingerprint")
    return fingerprint if isinstance(fingerprint, dict) else None


def newest_run(output_dir: Path) -> Path | None:
    if not output_dir.is_dir():
        return None
    runs = sorted(
        (entry for entry in output_dir.glob("run-*") if entry.is_dir()),
        key=lambda entry: entry.name,
    )
    return runs[-1] if runs else None


def kill_tree(process: subprocess.Popen) -> None:
    """Terminate only the queue we started, by process group."""
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        try:
            process.terminate()
        except ProcessLookupError:
            return
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def main() -> None:
    args = parse_args()
    for name in ("ae_root", "model_root", "libero_root", "out_root"):
        if not getattr(args, name):
            raise SystemExit(
                f"--{name.replace('_', '-')} or {name.upper()} must be set; no guessing"
            )

    expected = {}
    for stem in args.config_stem:
        path = Path(args.ae_root) / "configs" / "experiments" / f"{stem}.yaml"
        if not path.is_file():
            raise SystemExit(f"experiment config not found: {path}")
        expected[stem] = read_experiment(path)

    foreign_mib, foreign = gpu_foreign_mib(args.gpu)
    if foreign_mib > args.max_foreign_mib:
        raise SystemExit(
            f"GPU {args.gpu} is not usable: {foreign_mib} MiB held by other processes "
            f"(limit {args.max_foreign_mib}).\n  "
            + "\n  ".join(foreign)
            + "\nShared-card timings are worthless and the slowdown risks tripping "
            "predict_timeout_s, which would record winnable episodes as errors."
        )

    log_dir = Path(args.log_dir or (Path(args.out_root) / "_logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{args.lane}.log"
    env = dict(os.environ)
    env.update(
        AE_ROOT=args.ae_root,
        MODEL_ROOT=args.model_root,
        LIBERO_ROOT=args.libero_root,
        OUT_ROOT=args.out_root,
    )
    command = [
        "bash",
        "scripts/run-libero-queue.sh",
        args.lane,
        str(args.gpu),
        *args.config_stem,
    ]
    with log_path.open("ab") as log:
        process = subprocess.Popen(
            command,
            cwd=args.ae_root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    print(f"[launch] lane {args.lane} on GPU {args.gpu}, pid {process.pid}, log {log_path}")

    # Verify each configuration as soon as its worker has described itself.
    deadline = time.monotonic() + args.verify_timeout
    pending = dict(expected)
    verified: dict[str, Path] = {}
    try:
        while pending and time.monotonic() < deadline:
            for stem in list(pending):
                run_dir = newest_run(Path(args.out_root) / stem)
                if run_dir is None:
                    continue
                fingerprint = worker_fingerprint(run_dir)
                if fingerprint is None:
                    continue
                wanted = pending[stem]
                problems = []
                reported = fingerprint.get("sparse")
                if wanted["sparse"] is None:
                    if isinstance(reported, dict) and reported.get("enabled"):
                        problems.append(
                            f"config declares no sparse block but the worker reports "
                            f"{reported!r}"
                        )
                else:
                    if not isinstance(reported, dict):
                        problems.append(
                            "worker fingerprint has no `sparse` block: the running adapter "
                            "does not consume policy.options.sparse, so this run is DENSE"
                        )
                    else:
                        for key, value in wanted["sparse"].items():
                            if reported.get(key) != value:
                                problems.append(
                                    f"sparse.{key} is {reported.get(key)!r}, "
                                    f"expected {value!r}"
                                )
                if wanted["denoising_steps"] is not None:
                    reported_steps = fingerprint.get("denoising_steps")
                    if reported_steps != wanted["denoising_steps"]:
                        problems.append(
                            f"denoising_steps is {reported_steps!r}, "
                            f"expected {wanted['denoising_steps']!r}"
                        )
                if problems:
                    raise SystemExit(
                        f"[launch] REFUSING run {stem}: "
                        + "; ".join(problems)
                        + f"\n  run dir: {run_dir}"
                    )
                print(
                    f"[launch] verified {stem}: sparse={reported} "
                    f"config_hash={fingerprint.get('sparse_config_hash')}\n"
                    f"          run dir: {run_dir}"
                )
                verified[stem] = run_dir
                del pending[stem]
            if pending:
                time.sleep(10)
        if pending:
            raise SystemExit(
                f"[launch] timed out after {args.verify_timeout:.0f}s waiting for the "
                f"worker fingerprint of {sorted(pending)}; see {log_path}"
            )
    except BaseException:
        print("[launch] stopping the queue because verification failed", flush=True)
        kill_tree(process)
        raise

    print(
        json.dumps(
            {
                "lane": args.lane,
                "gpu": args.gpu,
                "verified": {stem: str(path) for stem, path in verified.items()},
                "pid": process.pid,
                "log": str(log_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    print("[launch] verification passed; the run continues in the background")


if __name__ == "__main__":
    main()
