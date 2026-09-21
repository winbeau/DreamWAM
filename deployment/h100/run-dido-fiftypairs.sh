#!/usr/bin/env bash
# Prepared CPU-rendered 50-pair run; release owned holds and launch in one shell.
set -euo pipefail
[[ $# -eq 2 ]] || { echo 'usage: run-dido-fiftypairs.sh FROZEN_EVAL_ROOT OUTPUT_ROOT' >&2; exit 2; }
DIDO_EVAL_TREE=$1
DIDO_OUT=$2
DIDO_CONTROLLER=$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
export DREAMWAM_SR_ROOT=/root/wenbiao_zhao/dreamwam-sr
export GPU_UUID=GPU-c0af33a9-498c-ff7c-bb56-e9992ccded30
source "$DIDO_CONTROLLER/deployment/h100/dido-sparse-profile.env.sh"
export MODEL_ROOT="$DREAMWAM_SR_ROOT/.trees/dido-run-41f515a-final"
export ACTION_EVAL_ROOT="$DIDO_EVAL_TREE"
export LIBERO_ROOT="$DREAMWAM_SR_ROOT/LIBERO"
export PYTHONPATH="$ACTION_EVAL_ROOT/src:$ACTION_EVAL_ROOT/packages/policy-sdk/src:$MODEL_ROOT:$LIBERO_ROOT"
DIDO_HOLD="$DREAMWAM_SR_ROOT/.trees/dido-hold-64471fb/scripts/sparse/gpu_hold.py"
DIDO_HOLD_STATE_3="$DREAMWAM_SR_ROOT/gpu-hold/h100-tenpairs-after-gpu3.json"
DIDO_HOLD_STATE_4="$DREAMWAM_SR_ROOT/gpu-hold/h100-tenpairs-after-gpu4.json"
DIDO_LEDGER="$DREAMWAM_SR_ROOT/outputs/dido-sparse-profile-20260920/closed-loop-ledger.json"
DIDO_ADAPTER="$DREAMWAM_SR_ROOT/outputs/dido-sparse-profile-20260920/adapter-41f515a-va56-shared50/report.json"
mkdir "$DIDO_OUT/handoff-once"
[[ $(git -C "$MODEL_ROOT" rev-parse HEAD) == 41f515a5480cdcc87a18b147ab393e9cae337e9f ]]
[[ -z $(git -C "$DIDO_CONTROLLER" status --porcelain) ]]
[[ -z $(git -C "$ACTION_EVAL_ROOT" status --porcelain) ]]
"$MODEL_PYTHON" - "$DIDO_HOLD" "$DIDO_HOLD_STATE_3" "$DIDO_HOLD_STATE_4" "$DIDO_LEDGER" <<'PY'
import importlib.util,json,sys
from pathlib import Path
spec=importlib.util.spec_from_file_location('hold',sys.argv[1])
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
workers=[w for path in sys.argv[2:4] for w in json.loads(Path(path).read_text())['workers']]
ledger=json.loads(Path(sys.argv[4]).read_text())
assert ledger['cap'] == 132 and sum(r['charged'] for r in ledger['reservations']) == 32
assert all(r['state'] == 'FINALIZED' for r in ledger['reservations'])
assert {w['gpu'] for w in workers} == {3,4}
assert all(module.owned(w) for w in workers), 'hold identity changed; nothing released'
PY

dido_rehold() {
  local gpu=$1 attempt
  for attempt in {1..30}; do
    if "$MODEL_PYTHON" "$DIDO_HOLD" start \
      --state "$DREAMWAM_SR_ROOT/gpu-hold/h100-fiftypairs-after-gpu${gpu}.json" \
      --gpus "$gpu" --minutes 120 >> "$DIDO_OUT/rehold-gpu${gpu}.log" 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

dido_lane() (
  local lane=$1 gpu=$2 uuid=$3 first=$4 rc
  export GPU_UUID="$uuid"
  set +e
  "$EVAL_PYTHON" "$DIDO_CONTROLLER/scripts/sparse/run_fresh_token_pair.py" \
    --eval-root "$ACTION_EVAL_ROOT" --model-root "$MODEL_ROOT" \
    --out-dir "$DIDO_OUT/lane-${lane}" --adapter-report "$DIDO_ADAPTER" \
    --dense-config "$ACTION_EVAL_ROOT/configs/experiments/dreamwam-dido-dense-h100-fiftypairs-${lane}.yaml" \
    --sparse-config "$ACTION_EVAL_ROOT/configs/experiments/dreamwam-dido-native-va56-h100-fiftypairs-${lane}.yaml" \
    --planned-episodes 25 --first-arm "$first" --authorized-gpus 3 4 5 \
    --render-backend osmesa --episode-ledger "$DIDO_LEDGER" --episode-cap 132 \
    --wall-seconds 0 --admission-seconds 120 --stop-grace-seconds 120 \
    > "$DIDO_OUT/lane-${lane}.launch.log" 2>&1
  rc=$?
  printf '%s\n' "$rc" > "$DIDO_OUT/lane-${lane}.launch.exit"
  dido_rehold "$gpu"
  printf '%s\n' "$?" > "$DIDO_OUT/rehold-gpu${gpu}.exit"
  exit "$rc"
)

# No manual gap, new dependency install, or preparation step after this release.
date -u +%FT%TZ > "$DIDO_OUT/handoff-start.txt"
"$MODEL_PYTHON" "$DIDO_HOLD" stop --state "$DIDO_HOLD_STATE_3" > "$DIDO_OUT/hold-release-gpu3.json"
"$MODEL_PYTHON" "$DIDO_HOLD" stop --state "$DIDO_HOLD_STATE_4" > "$DIDO_OUT/hold-release-gpu4.json"
dido_lane a 3 GPU-c0af33a9-498c-ff7c-bb56-e9992ccded30 dense &
DIDO_PID_A=$!
dido_lane b 4 GPU-3af22086-0ac4-4277-1486-96909b440d1c sparse &
DIDO_PID_B=$!
printf '%s %s\n' "$DIDO_PID_A" "$DIDO_PID_B" > "$DIDO_OUT/lanes.pid"
set +e
wait "$DIDO_PID_A"
DIDO_RC_A=$?
wait "$DIDO_PID_B"
DIDO_RC_B=$?
set -e
printf '%s %s\n' "$DIDO_RC_A" "$DIDO_RC_B" > "$DIDO_OUT/lanes.exit"
date -u +%FT%TZ > "$DIDO_OUT/handoff-end.txt"
"$EVAL_PYTHON" - "$DIDO_OUT" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
states={lane:json.loads((root/f'lane-{lane}/controller.json').read_text())['status'] for lane in ['a','b']}
print(json.dumps(states))
assert all(s == 'PILOT_PAIR_COMPLETE' for s in states.values()), 'incomplete cohort; inspect original errors, no automatic retries'
PY
