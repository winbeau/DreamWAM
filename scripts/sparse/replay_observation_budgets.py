#!/usr/bin/env python3
"""CPU-only M1 decisions on complete observed trajectories; no model/SR claim."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from observation_sequences import load_sequences, sha256
from dreamwam.sparse.chunk_budget import ChunkBudgetConfig, ObservationBudget
from dreamwam.sparse.hybrid import HybridConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--options", type=Path, default=Path("configs/sparse/m1-fixed-m2-m3.json"))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest, sequences = load_sequences(args.inputs)
    options = json.loads(args.options.read_text())
    config = ChunkBudgetConfig.from_mapping(options["chunk_budget"])
    hybrid = HybridConfig.from_mapping(options["hybrid_visual"])
    config.hybrid_configs(hybrid)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    report = dict(status="RUNNING", start_utc=datetime.now(timezone.utc).isoformat(), argv=sys.argv,
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        checkpoint_sha256=None, checkpoint_usage="not loaded; observation-only CPU controller replay",
        inputs_sha256=sha256(args.inputs), options_sha256=sha256(args.options),
        split=manifest["split_role"], source_episode_split_sha256=manifest["episode_split_sha256"],
        controller=config.describe(), controller_hash=config.policy_hash,
        frozen_hybrid=hybrid.describe(), frozen_hybrid_hash=hybrid.policy_hash,
        episodes=[], sr=None, limitations=["exposed development trajectories; not held-out confirmation",
            "uncalibrated observable-change rules are not causal action-sensitivity estimates",
            "no model inference, model latency, candidate closed-loop success or executed work measured"])
    try:
        for identity, sequence in sequences.items():
            controller = ObservationBudget(config)
            rows = []
            for entry, observation in sequence:
                decision = controller.propose(observation["images"], observation["state"])
                controller.commit(decision)
                rows.append(dict(input_id=entry["id"], input_sha256=entry["sha256"],
                    call_index=entry["call_index"], environment_step=entry["step_index"],
                    decision=controller.last_stats))
            label = "t%03d-i%03d-r%02d-s%d" % identity
            report["episodes"].append(dict(episode_id=label, calls=rows,
                level_counts=dict(Counter(row["decision"]["level"] for row in rows))))
            fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
            x = [row["environment_step"] for row in rows]
            axes[0].plot(x, [row["decision"]["score"] for row in rows], "o-", color="#4477aa")
            for threshold in config.thresholds:
                axes[0].axhline(threshold, color="#888888", ls="--", lw=1)
            axes[0].set_ylabel("Observable-change score")
            axes[0].set_title(label + " | raw RGB + proprio history | development")
            for name, color in (("query_ratio", "#228833"), ("read_ratio", "#aa3377")):
                axes[1].step(x, [row["decision"]["budget"][name] for row in rows],
                             where="post", label=name, color=color)
            axes[1].set_ylabel("Requested fraction")
            axes[1].set_ylim(0, 1)
            axes[1].legend(frameon=False)
            for name in dict(config.scales):
                axes[2].plot(x, [row["decision"]["normalized_features"][name] for row in rows],
                             label=name, lw=1.2)
            axes[2].set_ylabel("Normalized feature")
            axes[2].set_xlabel("Environment step (one point per observed chunk)")
            axes[2].legend(frameon=False, ncol=3, fontsize=8)
            fig.suptitle("M1 observation replay; frozen M2/M3; no model or SR measurement", fontsize=12)
            for extension in ("png", "pdf"):
                fig.savefig(args.out_dir / f"{label}.{extension}", dpi=170)
            plt.close(fig)
        costs = [row["decision"]["controller_seconds"] for episode in report["episodes"] for row in episode["calls"]]
        report.update(status="VERIFIED_OBSERVATION_REPLAY", calls=len(costs),
                      controller_seconds_mean=float(np.mean(costs)),
                      controller_seconds_max=float(np.max(costs)))
    except BaseException as exc:
        report.update(status="ERROR", error=repr(exc))
        raise
    finally:
        report["end_utc"] = datetime.now(timezone.utc).isoformat()
        report["artifacts"] = {path.name: sha256(path) for path in sorted(args.out_dir.iterdir()) if path.is_file()}
        (args.out_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "calls", "controller_seconds_mean", "sr")}))


if __name__ == "__main__":
    main()
