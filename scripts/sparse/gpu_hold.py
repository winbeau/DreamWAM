#!/usr/bin/env python3
"""Explicit, finite GPU hold. Stop signals only identity-checked owned workers."""
import argparse
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import time


def inventory():
    rows = {}
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,utilization.gpu,memory.free,memory.used',
                                      '--format=csv,noheader,nounits'], text=True, timeout=15)
    for line in output.splitlines():
        index, uuid, util, free, used = [v.strip() for v in line.split(',')]
        rows[int(index)] = dict(index=int(index), uuid=uuid, util=int(util), free=int(free), used=int(used))
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                   '--format=csv,noheader,nounits'], text=True, timeout=15)
    busy = {line.split(',')[0].strip() for line in apps.splitlines() if line.strip()}
    for row in rows.values():
        row['has_compute'] = row['uuid'] in busy
    return rows


def empty(row):
    return row['used'] <= 32 and row['util'] <= 10 and row['free'] >= 35000 and not row['has_compute']


def ticks(pid):
    return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]


def owned(row):
    try:
        cmd = Path(f"/proc/{row['pid']}/cmdline").read_bytes().split(b'\0')
        return (ticks(row['pid']) == row['start_ticks'] and row['token'].encode() in cmd
                and row['script'].encode() in cmd and b'--worker' in cmd)
    except (OSError, IndexError):
        return False


def write(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    os.replace(tmp, path)


def worker(args):
    running = True
    def finish(*_):
        nonlocal running
        running = False
    signal.signal(signal.SIGTERM, finish)
    signal.signal(signal.SIGINT, finish)
    record = dict(pid=os.getpid(), gpu=args.gpu, token=args.token, status='STARTING')
    try:
        row = inventory()[args.gpu]
        if not empty(row):
            raise RuntimeError(f'GPU is no longer empty: {row}')
        os.environ['CUDA_VISIBLE_DEVICES'] = row['uuid']
        import torch
        torch.set_num_threads(1)
        if not running:
            return
        torch.cuda.set_device(0)
        free, _ = torch.cuda.mem_get_info()
        target = min(int(free * args.fraction), free - 8 * 1024**3)
        matrix_bytes = 3 * args.matrix**2 * 2
        if target <= matrix_bytes + 1024**3:
            raise RuntimeError('insufficient headroom for requested matrices')
        with torch.inference_mode():
            reserve = torch.empty(target - matrix_bytes, dtype=torch.uint8, device='cuda')
            reserve.zero_()
            a = torch.full((args.matrix, args.matrix), 0.01, dtype=torch.bfloat16, device='cuda')
            b = torch.full_like(a, 0.01)
            c = torch.empty_like(a)
            torch.cuda.synchronize()
            record.update(status='HOLDING', uuid=row['uuid'], allocated_bytes=torch.cuda.memory_allocated(),
                          matrix=args.matrix, lease_seconds=args.seconds, ready_unix=time.time())
            write(args.ready, record)
            deadline = time.monotonic() + args.seconds
            while running and time.monotonic() < deadline:
                torch.mm(a, b, out=c)
                torch.cuda.synchronize()
        record.update(status='RELEASED', ended_unix=time.time())
    except BaseException as exc:
        record.update(status='ERROR', error=repr(exc), ended_unix=time.time())
        raise
    finally:
        write(args.ready, record)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['start', 'status', 'stop'], nargs='?', default='status')
    p.add_argument('--state', type=Path)
    p.add_argument('--gpus', type=int, nargs='+')
    p.add_argument('--minutes', type=int, default=60)
    p.add_argument('--fraction', type=float, default=0.8)
    p.add_argument('--matrix', type=int, default=32768)
    p.add_argument('--worker', action='store_true')
    p.add_argument('--gpu', type=int)
    p.add_argument('--token')
    p.add_argument('--ready', type=Path)
    p.add_argument('--seconds', type=int)
    args = p.parse_args()
    if args.worker:
        worker(args)
        return
    if args.state is None:
        p.error('--state is required')
    if args.command == 'start':
        if (not args.gpus or len(set(args.gpus)) != len(args.gpus) or
            not 1 <= args.minutes <= 120 or not 0.1 <= args.fraction <= 0.9 or
            not 1024 <= args.matrix <= 32768):
            p.error('explicit unique GPUs, 1..120 minutes, fraction 0.1..0.9 and matrix 1024..32768 required')
        if args.state.exists():
            old = json.loads(args.state.read_text())
            if any(owned(r) for r in old['workers']):
                p.error('owned workers already running; use status or stop')
            args.state.rename(args.state.with_name(args.state.name + f'.{time.time_ns()}.history'))
        args.state.parent.mkdir(parents=True, exist_ok=True)
        current = inventory()
        if any(i not in current or not empty(current[i]) for i in args.gpus):
            p.error('all requested GPUs must still be empty; no worker started')
        session = secrets.token_hex(8)
        folder = args.state.parent / session
        folder.mkdir()
        state = dict(status='STARTING', started_unix=time.time(), minutes=args.minutes,
                     admission=[current[i] for i in args.gpus], workers=[])
        script = str(Path(__file__).resolve())
        for gpu in args.gpus:
            token = secrets.token_hex(16)
            ready = folder / f'gpu{gpu}.json'
            cmd = [sys.executable, script, '--worker', '--gpu', str(gpu), '--token', token,
                   '--ready', str(ready), '--seconds', str(args.minutes * 60),
                   '--fraction', str(args.fraction), '--matrix', str(args.matrix)]
            with (folder / f'gpu{gpu}.log').open('w') as log:
                proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            state['workers'].append(dict(gpu=gpu, pid=proc.pid, start_ticks=ticks(proc.pid),
                                         token=token, script=script, ready=str(ready)))
            write(args.state, state)
        print(json.dumps(state))
        return
    state = json.loads(args.state.read_text())
    if args.command == 'stop':
        for row in state['workers']:
            if owned(row):
                os.kill(row['pid'], signal.SIGTERM)
        deadline = time.monotonic() + 5
        while any(owned(r) for r in state['workers']) and time.monotonic() < deadline:
            time.sleep(0.05)
        for row in state['workers']:
            if owned(row):
                os.kill(row['pid'], signal.SIGKILL)
        state.update(status='STOP_REQUESTED', stopped_unix=time.time())
        write(args.state, state)
    print(json.dumps([dict(gpu=r['gpu'], pid=r['pid'], owned_alive=owned(r),
                           report=json.loads(Path(r['ready']).read_text()) if Path(r['ready']).exists() else None)
                      for r in state['workers']], indent=2))


if __name__ == '__main__':
    main()
