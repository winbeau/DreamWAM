"""Append-only bounded raw-array archives; no pickle or silent overwrite."""

import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RawArchive:
    def __init__(self, root, *, max_bytes):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.max_bytes = max_bytes
        self.raw_bytes = 0
        self.records = []

    def write(self, input_id, kind, metadata, arrays):
        if not input_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in input_id):
            raise ValueError("unsafe input identity")
        size = sum(array.nbytes for array in arrays.values())
        if self.raw_bytes + size > self.max_bytes:
            raise RuntimeError("total raw archive byte budget exceeded")
        stem = f"{input_id}-{kind}-r{metadata['request']:03d}"
        if "step" in metadata:
            stem += f"-s{metadata['step']:02d}"
        if "layer" in metadata:
            stem += f"-l{metadata['layer']:02d}"
        path = self.root / (stem + ".npz")
        with path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        record = dict(input_id=input_id, kind=kind, **metadata, path=path.name,
            sha256=sha256(path), stored_bytes=path.stat().st_size, raw_bytes=size,
            arrays={name: dict(shape=list(value.shape), dtype=str(value.dtype))
                    for name, value in arrays.items()})
        with (self.root / "records.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
        self.records.append(record)
        self.raw_bytes += size
        return record
