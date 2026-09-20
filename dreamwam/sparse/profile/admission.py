"""Read-only admission for this H100 cohort; no reservation or process signals."""

import csv
import io
import subprocess


def eligible_devices(csv_text, authorized=(3, 4, 5)):
    rows = list(csv.reader(io.StringIO(csv_text)))
    devices = []
    for row in rows:
        if len(row) != 6:
            raise ValueError("invalid GPU inventory")
        index, uuid, name, used, total, util = (v.strip() for v in row)
        index, used, total, util = int(index), int(used), int(total), int(util)
        if index in authorized and "H100" in name and used <= 1024 and util <= 10 and total - used >= 50000:
            devices.append(dict(index=index, uuid=uuid, used_mib=used, total_mib=total, utilization=util))
    return devices


def admit_profile(uuid, *, authorized=(3, 4, 5)):
    command = ["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
               "--format=csv,noheader,nounits"]
    text = subprocess.check_output(command, text=True)
    eligible = eligible_devices(text, authorized)
    if len(eligible) < 2:
        raise RuntimeError("need two available authorized H100 cards; leave the last one unused")
    if uuid not in [d["uuid"] for d in eligible]:
        raise RuntimeError("selected GPU is not currently admitted")
    return dict(command=command, inventory=text, eligible=eligible, selected=uuid,
                shared_host=True, selected_gpu_sharing=False, reservation=False)
