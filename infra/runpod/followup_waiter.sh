#!/usr/bin/env bash
# Laptop side of a follow-up stage run: serves the monitor relay, keeps the SSH reverse tunnel up,
# waits for the job's exit.code (or a watchdog teardown), then pulls results from HF, deletes the
# Pod and lists Pods. Usage: infra/runpod/followup_waiter.sh <pod-name> <SA|SB>
set -uo pipefail
NAME=$1 ARM=$2
ROOT=$(cd "$(dirname "$0")/../.." && pwd); cd "$ROOT"
OUT=local/results/followup/$ARM; mkdir -p $OUT
set -a; . ./.env; set +a
export PYTHONPATH=$ROOT
ts() { date '+%F %T'; }
POD=infra/runpod/graft_pod.py
SSH=$(python3 -c "import sys,shlex; sys.path.insert(0,'infra/runpod'); import graft_pod as g; print(shlex.join(g.ssh_argv(g.load('$NAME'))))")
deleted() { python3 -c "import sys; sys.path.insert(0,'infra/runpod'); import graft_pod as g; sys.exit(0 if g.load('$NAME').get('deleted_at') else 1)"; }

python3 -m monitors.relay --config configs/monitor/frozen_cot_v1.yaml --cache $OUT/relay_cache.sqlite \
  --port 8200 --workers 48 >> $OUT/relay.log 2>&1 &
RELAY=$!
( while true; do
    $SSH -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=4 \
      -R 8200:127.0.0.1:8200 >> $OUT/tunnel.log 2>&1
    echo "$(ts) tunnel exited, reconnecting" >> $OUT/tunnel.log; sleep 5
  done ) &
TUNNEL=$!
caffeinate -i -s -w $$ &
echo "$(ts) waiter up: relay $RELAY tunnel $TUNNEL"

while true; do
  sleep 120
  if deleted; then echo "$(ts) Pod already deleted (watchdog)"; break; fi
  r=$(python3 $POD job $NAME $ARM --lines 2 2>/dev/null | tail -1)
  case "$r" in [0-9]*) echo "$(ts) job exit $r"; break ;; esac
  if ! kill -0 $RELAY 2>/dev/null; then
    echo "$(ts) relay died, restarting"
    python3 -m monitors.relay --config configs/monitor/frozen_cot_v1.yaml --cache $OUT/relay_cache.sqlite \
      --port 8200 --workers 48 >> $OUT/relay.log 2>&1 &
    RELAY=$!
  fi
done
python3 $POD job $NAME $ARM --lines 60 > $OUT/job_tail.log 2>&1 || true
kill $TUNNEL $RELAY 2>/dev/null; pkill -f "8200:127.0.0.1:8200" 2>/dev/null
python3 - <<PY || echo "HF pull failed"
from huggingface_hub import snapshot_download
p = snapshot_download("ic-org/graft-monitors-pilot-runs", repo_type="dataset",
                      allow_patterns=["followup/$ARM/*"], local_dir="$OUT/hf")
print("pulled to", p)
PY
deleted || python3 $POD down $NAME
python3 infra/runpod/runpod_cli.py pods
echo "$(ts) waiter done"
