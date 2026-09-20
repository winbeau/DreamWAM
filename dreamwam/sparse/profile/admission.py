"""Read-only admission for this H100 cohort; no reservation or process signals."""

import csv
from datetime import datetime, timezone
import io
import subprocess


def eligible_devices(csv_text, authorized=(3, 4, 5), *, share_gpu5=False):
    rows = list(csv.reader(io.StringIO(csv_text)))
    devices = []
    for row in rows:
        if len(row) != 6:
            raise ValueError("invalid GPU inventory")
        index, uuid, name, used, total, util = (v.strip() for v in row)
        index, used, total, util = int(index), int(used), int(total), int(util)
        memory_ok = used <= 1024 or (share_gpu5 and index == 5)
        if index in authorized and "H100" in name and memory_ok and util <= 10 and total - used >= 50000:
            devices.append(dict(index=index, uuid=uuid, used_mib=used, total_mib=total, utilization=util))
    return devices


def admit_profile(uuid, *, authorized=(3, 4, 5), share_gpu5=False):
    command = ["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
               "--format=csv,noheader,nounits"]
    text = subprocess.check_output(command, text=True)
    eligible = eligible_devices(text, authorized, share_gpu5=share_gpu5)
    selected = next((device for device in eligible if device["uuid"] == uuid), None)
    sharing_exception = bool(share_gpu5 and selected and selected["index"] == 5)
    if not sharing_exception and len(eligible_devices(text, authorized)) < 2:
        raise RuntimeError("need two available authorized H100 cards; leave the last one unused")
    if uuid not in [d["uuid"] for d in eligible]:
        raise RuntimeError("selected GPU is not currently admitted")
    return dict(command=command, inventory=text, eligible=eligible, selected=uuid,
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                shared_host=True, selected_gpu_sharing=sharing_exception, reservation=False,
                minimum_free_mib=50000, maximum_utilization=10,
                authorization="2026-09-20 explicit flexible GPU 5 sharing" if sharing_exception else "standard admission",
                last_available_exception=sharing_exception)
