#!/usr/bin/env python3
"""Fit only M1 change-score quantiles on development observations, without outcomes."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from dreamwam.sparse.chunk_budget import ChunkBudgetConfig, ObservationBudget
from observation_sequences import load_sequences, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--options", type=Path, default=Path("configs/sparse/m1-fixed-m2-m3.json"))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest, sequences = load_sequences(args.inputs)
    if manifest["split_role"] != "development":
        raise ValueError("calibration may not consume confirmation observations")
    options = json.loads(args.options.read_text())
    config = ChunkBudgetConfig.from_mapping(options["chunk_budget"])
    scores, rows = [], []
    for identity, sequence in sequences.items():
        controller = ObservationBudget(config)
        for entry, observation in sequence:
            value = controller.propose(observation["images"], observation["state"])
            controller.commit(value)
            eligible = bool(value.diagnostics["history_length"])
            rows.append(dict(input_id=entry["id"], input_sha256=entry["sha256"],
                             score=value.diagnostics["score"], used_for_quantiles=eligible))
            if eligible: scores.append(value.diagnostics["score"])
    if len(scores) < 4:
        raise ValueError("insufficient non-bootstrap development observations")
    quantiles = np.arange(1, len(config.levels)) / len(config.levels)
    thresholds = np.quantile(scores, quantiles).tolist()
    options["chunk_budget"]["thresholds"] = thresholds
    calibrated = ChunkBudgetConfig.from_mapping(options["chunk_budget"])
    args.out_dir.mkdir(parents=True, exist_ok=False)
    destination = args.out_dir / "proposed-options.json"
    destination.write_text(json.dumps(options, indent=2, allow_nan=False) + "\n")
    report = dict(status="DEVELOPMENT_QUANTILES_PROPOSED", utc=datetime.now(timezone.utc).isoformat(),
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), argv=sys.argv,
        checkpoint_sha256=None, checkpoint_usage="not loaded", exit_code=0,
        inputs_sha256=sha256(args.inputs), input_split_sha256=manifest["episode_split_sha256"],
        original_options_sha256=sha256(args.options), proposed_options_sha256=sha256(destination),
        quantiles=quantiles.tolist(), thresholds=thresholds, controller_hash=calibrated.policy_hash,
        non_bootstrap_calls=len(scores), inputs=rows, sr=None,
        limitations=["observation-distribution calibration only; no action sensitivity or success optimization",
                     "three development episodes cannot establish generalization; independent confirmation remains required",
                     "hysteresis means executed budget frequencies need not match quantile frequencies"],
        next_step="review and commit the proposed thresholds locally before model execution; verify on separate observations")
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "thresholds", "non_bootstrap_calls", "sr")}))


if __name__ == "__main__":
    main()
