# Follow-up Stage A/B: one reasoning-monitor run per H200 Pod. Env: GM_ARM=SA|SB.
# The monitor is served from the laptop (monitors/relay.py) through an SSH reverse tunnel on
# 127.0.0.1:8200, so no monitor key is on this Pod. Rollouts go to HF every 20 min while training.
set -euo pipefail
: "${GM_ARM:?}"
case $GM_ARM in SA|SB) ;; *) echo "GM_ARM must be SA or SB"; exit 2 ;; esac
source /workspace/code/infra/runpod/jobs/bootstrap.sh
test -f $CODE/configs/monitor/frozen_cot_v1.yaml || { echo "no frozen monitor config"; exit 2; }
OUTR=$WS/runs/followup_$GM_ARM; mkdir -p $WS/telemetry $WS/evals
HFP=followup/$GM_ARM
for i in $(seq 1 60); do
  curl -sf http://127.0.0.1:8200/describe > $WS/telemetry/${GM_ARM}_relay.json && break
  [ $i = 60 ] && { echo "monitor relay not reachable after 10 min"; exit 3; }
  sleep 10
done
stamp "relay: $(cat $WS/telemetry/${GM_ARM}_relay.json)"
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 15 > $WS/telemetry/${GM_ARM}_gpu.csv &
SMI_PID=$!
push() {  # push(src_dir_or_file, path_in_repo)
  /opt/venvs/graft/bin/python - "$1" "$2" <<'PY' || echo "HF push failed (will retry next round)"
import os, sys
from huggingface_hub import HfApi
api, (src, dst) = HfApi(), sys.argv[1:3]
repo = "ic-org/graft-monitors-pilot-runs"
api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
if os.path.isdir(src):
    api.upload_folder(folder_path=src, repo_id=repo, repo_type="dataset", path_in_repo=dst,
                      ignore_patterns=["*/optimizer.pt", "*/scheduler.pt", "*.safetensors", "*/rng_state*"])
else:
    api.upload_file(path_or_fileobj=src, path_in_repo=dst, repo_id=repo, repo_type="dataset")
print("pushed", dst)
PY
}
( while sleep 1200; do
    R=$(ls -d $OUTR*/gm_${GM_ARM}_* 2>/dev/null | tail -1)
    [ -n "$R" ] && push "$R" "$HFP/run"
    push $WS/telemetry/${GM_ARM}_gpu.csv "$HFP/${GM_ARM}_gpu.csv"
  done ) &
PUSH_PID=$!
source $GRAFT_VENV/bin/activate
cd $GRAFT_RL
export GRAFT_ROLLOUT_LOG_DIR=auto GRAFT_ARM=$GM_ARM
stamp "Stage $GM_ARM start"
python -u sweep.py --name gm_followup_$GM_ARM --config sweeps/gm_phase3.py --no_baseline --no_mps \
    --output_dir $OUTR --wandb_project graft-dm-pilot ${EXTRA_TRAIN_ARGS:-}
stamp "Stage $GM_ARM training done"
kill $PUSH_PID 2>/dev/null || true
RUN=$(ls -d $OUTR*/gm_${GM_ARM}_* | tail -1)
push "$RUN" "$HFP/run"
cd /tmp
python -u $CODE/scripts/posthoc_eval.py --base $SFT_MODEL --run_dir $RUN --steps 0,50,100,150,200 \
    --out $WS/evals/${GM_ARM}.parquet
stamp "Stage $GM_ARM post-hoc eval done"
kill $SMI_PID || true
python -m analysis.metrics --run_dir $RUN --eval $WS/evals/${GM_ARM}.parquet --out $WS/evals/${GM_ARM}.metrics.json || true
for f in $WS/evals/${GM_ARM}.parquet $WS/evals/${GM_ARM}.meta.json $WS/evals/${GM_ARM}.metrics.json \
         $WS/telemetry/${GM_ARM}_gpu.csv $WS/telemetry/${GM_ARM}_relay.json; do
  [ -f $f ] && push $f "$HFP/$(basename $f)"
done
push "$RUN" "$HFP/run"
stamp "followup $GM_ARM done"
