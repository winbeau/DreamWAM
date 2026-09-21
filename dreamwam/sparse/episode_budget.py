"""Conservative, process-safe episode reservations for one bounded effort.

Reserve each pair's entire declared manifest before spawning either evaluator.
Charges are monotone: errors, interrupted runs, and missing terminal artifacts
never silently return budget. Recorded attempts are evidence, not a claim that
every reserved episode ran. A reservation is not proof of a live process.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path


def utc():
    return datetime.now(timezone.utc).isoformat()


class EpisodeBudget:
    def __init__(self, path, *, cap=50, effort="dido-sparse-profile-20260920"):
        if type(cap) is not int or cap < 1:
            raise ValueError("episode cap must be a positive integer")
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
                amendments = payload.get("cap_amendments", [])
                if self.cap > 50 and not amendments:
                    raise ValueError("expanded cap requires a recorded authorization amendment")
                if amendments:
                    previous = amendments[0]["from_cap"]
                    if not 1 <= previous <= 50:
                        raise ValueError("invalid original episode cap")
                    for amendment in amendments:
                        if (amendment["from_cap"] != previous or amendment["to_cap"] <= previous
                            or not amendment["authorization"].strip()
                            or len(amendment["previous_ledger_sha256"]) != 64):
                            raise ValueError("invalid cap amendment history")
                        previous = amendment["to_cap"]
                    if previous != self.cap:
                        raise ValueError("amendment history does not reach current cap")
                if (len({entry["id"] for entry in entries}) != len(entries) or
                    any(type(entry["charged"]) is not int or entry["charged"] <= 0 for entry in entries) or
                    sum(entry["charged"] for entry in entries) > self.cap):
                    raise ValueError("invalid episode reservation history")
            else:
                if self.cap > 50:
                    raise ValueError("expanded cap requires an existing conserved ledger")
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

    def amend_cap(self, new_cap, *, authorization, expected_charged):
        """Explicit user scope extension; retain every reservation and a hash chain."""
        if type(new_cap) is not int or new_cap <= self.cap:
            raise ValueError("amended cap must increase the existing cap")
        if not isinstance(authorization, str) or not authorization.strip():
            raise ValueError("cap amendment requires the explicit user authorization")
        with self.locked() as payload:
            if not self.path.exists():
                raise ValueError("amend an existing ledger, never replace its history")
            charged = sum(entry["charged"] for entry in payload["reservations"])
            if charged != expected_charged or any(e["state"] != "FINALIZED" for e in payload["reservations"]):
                raise ValueError("cap amendment requires the expected finalized history")
            amendment = dict(from_cap=self.cap, to_cap=new_cap, utc=utc(),
                authorization=authorization, total_charged=charged,
                previous_ledger_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest())
            payload.setdefault("cap_amendments", []).append(amendment)
            payload["cap"] = new_cap
            self._write(payload)
            self.cap = new_cap
            return amendment

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
