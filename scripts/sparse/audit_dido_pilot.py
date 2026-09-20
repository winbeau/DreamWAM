#!/usr/bin/env python3
"""Audit a complete bounded pair's actual actions, native reads and budget receipt."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from paired_sr import load_manifest


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root):
    controller = json.loads((root / "controller.json").read_text())
    if controller["status"] != "PILOT_PAIR_COMPLETE":
        raise ValueError("require a complete pilot; incomplete coverage has no full SR")
    receipt = controller["episode_budget"]
    if not receipt or receipt["cap"] != 50 or receipt["total_charged"] > 50:
        raise ValueError("missing or invalid effort budget receipt")
    result, manifests, total_attempts = {}, [], 0
    checkpoint = None
    for arm in ("dense", "sparse"):
        directory = root / arm / "run"
        manifest, planned = load_manifest(directory)
        manifests.append(manifest)
        if len(planned) != controller["planned_episodes_per_arm"]:
            raise ValueError("manifest differs from the reserved pilot")
        protocol = manifest["protocol"]
        if protocol["id"] != "dreamwam-release-v1" or protocol["replan_steps"] != 10:
            raise ValueError("release protocol changed")
        provenance = json.loads((directory / "provenance.json").read_text())
        fingerprint = provenance["policy_worker"]["description"]["fingerprint"]
        digest = fingerprint["checkpoint_sha256"]
        checkpoint = digest if checkpoint is None else checkpoint
        if digest != checkpoint or fingerprint["action_horizon"] != 32 or fingerprint["denoising_steps"] != 10:
            raise ValueError("checkpoint/action sampling differs between arms")
        episodes, seen = [], set()
        for path in sorted(directory.glob("episodes/*/attempts/*/result.json")):
            row = json.loads(path.read_text())
            identity = tuple(row[key] for key in ("task_id", "init_index", "repeat", "seed"))
            if identity not in planned or identity in seen or row["attempt"] != 1:
                raise ValueError("unplanned, duplicate or retried episode")
            seen.add(identity)
            if row["status"] not in ("succeeded", "failed") or row["success"] != (row["status"] == "succeeded"):
                raise ValueError("infrastructure error or contradictory outcome is not a task result")
            arrays = {key: np.load(directory / row["artifacts"][key], allow_pickle=False)
                      for key in ("actions", "predictions", "prediction_lengths")}
            actions, predictions, lengths = (arrays[key] for key in ("actions", "predictions", "prediction_lengths"))
            if (actions.shape != (row["executed_steps"], 7) or actions.dtype != np.float32 or
                predictions.shape != (32 * row["policy_calls"], 7) or predictions.dtype != np.float32 or
                lengths.shape != (row["policy_calls"],) or not np.all(lengths == 32) or
                not np.isfinite(actions).all() or not np.isfinite(predictions).all()):
                raise ValueError("archived action horizon, shape, dtype or values changed")
            executed = predictions.reshape(-1, 32, 7)[:, :10].reshape(-1, 7)[:len(actions)]
            if executed.tobytes() != actions.tobytes():
                raise ValueError("actual simulator actions differ from the predicted execution prefixes")
            calls = json.loads((directory / row["artifacts"]["policy_calls"]).read_text())
            if len(calls) != row["policy_calls"]:
                raise ValueError("missing per-call evidence")
            fallback = 0
            for call in calls:
                stats = call["metadata"]["diagnostics"]["hybrid_visual" if arm == "sparse" else "visual_cache"]
                if stats["action_layer_updates"] != 300:
                    raise ValueError("action-layer execution budget changed")
                if arm == "sparse":
                    if (stats["computed_video_token_layers"] != 294 * 30 or
                        stats["read_video_token_layers"] != (294 + 9 * 56) * 30 or
                        [step["effective_op"] for step in stats["steps"]] != ["dense"] + ["reuse"] * 9 or
                        any(step["read_layout"] != "layerwise" for step in stats["steps"])):
                        raise ValueError("frozen native layerwise R56 execution changed")
                    fallback += stats["selector_fallback_events"]
            episodes.append(dict(episode_id=row["episode_id"], status=row["status"],
                executed_steps=len(actions), policy_calls=len(calls), fallback_events=fallback,
                actual_actions_match_prediction_prefixes=True, result_sha256=sha256(path)))
        if seen != planned:
            raise ValueError("incomplete planned episode coverage")
        native_reads = 0
        logs = list((root / arm).glob("native-reads-*.log"))
        if len(logs) != 1:
            raise ValueError("expected one native read journal per bounded arm")
        lines = logs[0].read_text().splitlines()
        if not lines or len(lines) % 2:
            raise ValueError("unfinished native read")
        for offset in range(0, len(lines), 2):
            index = offset // 2 + 1
            if lines[offset:offset + 2] != [f"{index} start", f"{index} done"]:
                raise ValueError("missing, failed or reordered native RGB read")
            native_reads += 1
        result[arm] = dict(episodes=episodes, native_reads=native_reads)
        total_attempts += len(episodes)
    if (manifests[0]["benchmark"] != manifests[1]["benchmark"] or
        manifests[0]["protocol"] != manifests[1]["protocol"] or total_attempts != receipt["charged"]):
        raise ValueError("pair identities/protocol or episode reservation differ")
    artifacts = {str(path.relative_to(root)): sha256(path) for path in sorted(root.rglob("*")) if path.is_file()}
    return dict(status="VERIFIED", source_model_commit=controller["model_commit"],
        evaluator_commit=controller["evaluator_commit"], controller_source_sha256=controller["controller_source_sha256"],
        checkpoint_sha256=checkpoint, recorded_attempts=total_attempts, receipt=receipt,
        arms=result, artifacts=artifacts, sr_source="benchmark-owned terminal records; paired_sr.py reports rates/intervals")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.out.resolve().is_relative_to(args.pair.resolve()):
        parser.error("use a new audit output outside the immutable pair tree")
    start = datetime.now(timezone.utc).isoformat()
    result = audit(args.pair)
    result.update(start_utc=start, end_utc=datetime.now(timezone.utc).isoformat(), exit_code=0,
        argv=sys.argv, auditor_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
    with args.out.open("x") as handle:
        handle.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(status=result["status"], recorded_attempts=result["recorded_attempts"],
                         verified_artifacts=len(result["artifacts"]))))


if __name__ == "__main__":
    main()
