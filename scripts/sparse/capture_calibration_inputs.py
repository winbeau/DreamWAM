#!/usr/bin/env python3
"""Capture fixed post-settle LIBERO observations for offline M1 replay.

Runs no policy, executes no evaluated episode and reports no success rate.
The existing evaluator supplies the protocol, images, proprio and instruction.
These official identities become exposed calibration inputs, not held-out SR.
Run in a separate renderer process before loading any CUDA model on its GPU.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--tasks", type=int, nargs="+", default=[0, 4, 8])
    parser.add_argument("--init-index", type=int, default=0)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    from action_eval.config import load_experiment
    from action_eval.benchmarks.libero.backend import LiberoBenchmark
    loaded = load_experiment(args.config, require_paths=True)
    protocol = loaded.config.require_protocol()
    if len(set(args.tasks)) != len(args.tasks):
        parser.error("duplicate tasks")
    if any(t not in loaded.config.benchmark.task_ids for t in args.tasks):
        parser.error("task outside the declared benchmark")
    if args.init_index not in loaded.config.benchmark.initial_state_indices:
        parser.error("initial state outside the declared benchmark")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    manifest = dict(status="RUNNING", start_utc=datetime.now(timezone.utc).isoformat(),
                    purpose="offline input capture; no policy, no task outcome, not held-out SR",
                    git=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    config_source=args.config, config_sha256=sha(args.config),
                    config_fingerprint=loaded.fingerprint, protocol=loaded.resolved["protocol"],
                    suite=loaded.config.benchmark.suite,
                    cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), inputs=[], sr=None)
    write = lambda: (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write()
    benchmark = LiberoBenchmark(loaded)
    try:
        for task in args.tasks:
            environment = benchmark.make_environment(task_id=task)
            try:
                environment.reset(task_id=task, init_index=args.init_index)
                observation = environment.wait(protocol.wait_steps)
                name = f"t{task:03d}-i{args.init_index:03d}"
                path = out / f"{name}.npz"
                np.savez_compressed(path, agentview=observation.images["agentview"],
                    wrist=observation.images["wrist"], state=observation.state,
                    instruction=np.asarray(observation.instruction))
                entry = dict(id=name, task_id=task, init_index=args.init_index,
                    task_name=benchmark.task_name(task), seed=protocol.seed,
                    step_index=observation.step_index, instruction=observation.instruction,
                    path=path.name, sha256=sha(path), state=observation.state.tolist(),
                    images={k:dict(shape=list(v.shape), dtype=str(v.dtype),
                        sha256=hashlib.sha256(v.tobytes()).hexdigest(),
                        minimum=int(v.min()), maximum=int(v.max()), stdev=float(v.std()))
                        for k,v in observation.images.items()})
                manifest["inputs"].append(entry)
                write()
                print(json.dumps(entry), flush=True)
            finally:
                environment.close()
        manifest.update(status="CAPTURED", end_utc=datetime.now(timezone.utc).isoformat())
        write()
    except BaseException as exc:
        manifest.update(status="ERROR", error=repr(exc), end_utc=datetime.now(timezone.utc).isoformat())
        write()
        raise
    finally:
        benchmark.close()


if __name__ == "__main__":
    main()
