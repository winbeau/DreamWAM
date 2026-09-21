"""Hash-checked observation sequences, without outcomes or teacher predictions."""

from collections import defaultdict
import hashlib
import json

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_sequences(manifest_path):
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "CAPTURED" or manifest.get("complete_call_history") is not True:
        raise ValueError("M1 requires a captured complete-call-history manifest, not sparse snapshots")
    if manifest.get("split_role") not in ("development", "confirmation"):
        raise ValueError("observation split must be explicitly development or confirmation")
    if not manifest.get("episode_split_sha256") or not manifest.get("inputs"):
        raise ValueError("missing episode split identity or observations")
    groups, identities = defaultdict(list), set()
    for entry in manifest["inputs"]:
        if entry["id"] in identities:
            raise ValueError("duplicate observation identity")
        identities.add(entry["id"])
        identity = tuple(entry[key] for key in ("task_id", "init_index", "repeat", "seed"))
        path = manifest_path.parent / entry["path"]
        if sha256(path) != entry["sha256"]:
            raise ValueError("captured observation hash mismatch")
        with np.load(path, allow_pickle=False) as values:
            if set(values) != {"state", "instruction", "agentview", "wrist"}:
                raise ValueError("only policy-observable inputs are allowed")
            observation = dict(images={k: values[k].copy() for k in ("agentview", "wrist")},
                state=values["state"].copy(), instruction=str(values["instruction"].item()))
        groups[identity].append((entry, observation))
    for sequence in groups.values():
        sequence.sort(key=lambda pair: pair[0]["call_index"])
        if [entry["call_index"] for entry, _ in sequence] != list(range(1, len(sequence) + 1)):
            raise ValueError("history has missing or duplicate calls")
        steps = [entry["step_index"] for entry, _ in sequence]
        if any(a >= b for a, b in zip(steps, steps[1:])):
            raise ValueError("observation environment steps must strictly increase")
    return manifest, dict(sorted(groups.items()))
