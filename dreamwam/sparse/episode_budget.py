"""Conservative, process-safe episode reservations for one bounded effort.

Reserve each pair's entire declared manifest before spawning either evaluator.
Charges are monotone: errors, interrupted runs, and missing terminal artifacts
never silently return budget. Recorded attempts are evidence, not a claim that
every reserved episode ran. A reservation is not proof of a live process.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path


def utc():
    return datetime.now(timezone.utc).isoformat()


class EpisodeBudget:
    def __init__(self, path, *, cap=50, effort="dido-sparse-profile-20260920"):
        if type(cap) is not int or not 1 <= cap <= 50:
            raise ValueError("episode cap must be an integer from 1 to 50")
        self.path = Path(path).resolve()
        self.cap, self.effort = cap, effort

    @contextmanager
    def locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix(self.path.suffix + ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if self.path.exists():
                payload = json.loads(self.path.read_text())
                if (payload.get("schema_version") != 1 or payload.get("cap") != self.cap or
                    payload.get("effort") != self.effort):
                    raise ValueError("existing episode ledger identity/cap differs")
                entries = payload["reservations"]
                if (len({entry["id"] for entry in entries}) != len(entries) or
                    any(type(entry["charged"]) is not int or entry["charged"] <= 0 for entry in entries) or
                    sum(entry["charged"] for entry in entries) > self.cap):
                    raise ValueError("invalid episode reservation history")
            else:
                payload = dict(schema_version=1, effort=self.effort, cap=self.cap,
                               created_utc=utc(), reservations=[])
            yield payload

    def _write(self, payload):
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary.open("w") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.path)
        directory = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    def available(self):
        with self.locked() as payload:
            return self.cap - sum(entry["charged"] for entry in payload["reservations"])

    def reserve(self, reservation_id, count, *, metadata):
        if not isinstance(reservation_id, str) or not reservation_id or type(count) is not int or count <= 0:
            raise ValueError("episode reservation requires a unique name and positive integer count")
        with self.locked() as payload:
            if any(entry["id"] == reservation_id for entry in payload["reservations"]):
                raise ValueError("episode reservation already exists; no automatic retry")
            remaining = self.cap - sum(entry["charged"] for entry in payload["reservations"])
            if count > remaining:
                raise ValueError(f"episode budget exhausted: requested {count}, remaining {remaining}, cap {self.cap}")
            entry = dict(id=reservation_id, charged=count, created_utc=utc(),
                         state="RESERVED", metadata=metadata, evidence=None)
            payload["reservations"].append(entry)
            self._write(payload)
            return dict(path=str(self.path), reservation_id=reservation_id, charged=count,
                        total_charged=self.cap - remaining + count, cap=self.cap)

    def finish(self, reservation_id, *, evidence):
        with self.locked() as payload:
            entry = next(row for row in payload["reservations"] if row["id"] == reservation_id)
            if entry["state"] != "RESERVED":
                raise ValueError("episode reservation already finalized")
            entry.update(state="FINALIZED", finalized_utc=utc(), evidence=evidence)
            self._write(payload)
            return dict(path=str(self.path), reservation_id=reservation_id, charged=entry["charged"],
                        total_charged=sum(row["charged"] for row in payload["reservations"]), cap=self.cap,
                        evidence=evidence)
