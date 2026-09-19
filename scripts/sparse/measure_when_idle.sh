#!/bin/bash
# Wait for an idle GPU, then measure the routed path at full request level.
#
# Why a waiter instead of a launch: this project shares the machine with other users, and a
# contended card both invalidates timings and risks tripping the evaluator's predict timeout.
# The loop refuses to start until the card is genuinely idle, then runs the same weights
# through dense and through the sparse variants so the amortization question is answered by
# measurement rather than by argument.
#
# usage: measure_when_idle.sh <physical-gpu> <output-dir> [max-wait-minutes]
set -u

GPU="${1:?physical GPU index required}"
OUT="${2:?output directory required}"
MAX_WAIT_MIN="${3:-240}"
POLL_SECONDS=60

mkdir -p "$OUT"
deadline=$(( $(date +%s) + MAX_WAIT_MIN * 60 ))

echo "[wait] $(date -Is) waiting for GPU $GPU to become idle (up to ${MAX_WAIT_MIN} min)"

while :; do
  used="$(nvidia-smi -i "$GPU" --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | paste -sd+ - | bc 2>/dev/null)"
  used="${used:-0}"
  if [ "$used" -le 0 ] 2>/dev/null; then
    echo "[wait] $(date -Is) GPU $GPU is idle; starting measurements"
    break
  fi
  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "[wait] $(date -Is) giving up after ${MAX_WAIT_MIN} min; GPU $GPU still holds ${used} MiB"
    exit 3
  fi
  sleep "$POLL_SECONDS"
done

run_variant() {
  local label="$1"; shift
  local extra=("$@")
  echo "[measure] $label"
  CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python scripts/sparse/profile_dense.py \
    --config configs/dreamwam_joint.yaml \
    --warmup 2 --reps 8 --no-instrument \
    "${extra[@]}" \
    --out "$OUT/$label.json" >"$OUT/$label.log" 2>&1
  local status=$?
  if [ $status -ne 0 ]; then
    echo "[measure] $label FAILED (exit $status); see $OUT/$label.log"
    return $status
  fi
  python3 - "$OUT/$label.json" "$label" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
latency = payload["request_latency_seconds"]
sparse = payload.get("sparse") or {}
print(f"  {sys.argv[2]:<34} p50={latency['p50']:.4f}s mean={latency['mean']:.4f}s "
      f"sparse={sparse.get('enabled')} refresh={sparse.get('anchor_refresh')} "
      f"ratio={sparse.get('future_ratio')}")
PY
}

run_variant dense
for refresh in layer step request; do
  run_variant "sparse_av25_${refresh}" \
    --sparse-json "{\"enabled\": true, \"selection\": \"av\", \"backend\": \"masked\", \"block_size\": 14, \"future_ratio\": 0.25, \"anchor_refresh\": \"${refresh}\"}"
done

echo "[measure] $(date -Is) done; results in $OUT"
