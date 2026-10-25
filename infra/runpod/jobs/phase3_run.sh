# Phase 3: one pilot run per H200 Pod. Env: GM_ARM in P1..P4 (P5 later). For decision arms the
# frozen monitor config must exist (configs/monitor/frozen_v1.yaml) and, if local, AnyJev is served
# here with AJ_MODEL / AJ_KIND / AJ_MEM, on GPU AJ_GPU if set (training then stays on GPU 0).
set -euo pipefail
: "${GM_ARM:?}"
case $GM_ARM in P3|P4|P5) DECISION=1 ;; *) DECISION=0 ;; esac
[ $DECISION = 1 ] && [ "${AJ_MODEL:-}" != "" ] && export NEED_ANYJEV=1
source /workspace/code/infra/runpod/jobs/bootstrap.sh
OUTR=$WS/runs/phase3_$GM_ARM; mkdir -p $WS/telemetry $WS/evals   # sweep.py suffixes an existing output_dir
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 15 > $WS/telemetry/${GM_ARM}_gpu.csv &
SMI_PID=$!
if [ $DECISION = 1 ]; then
  test -f $CODE/configs/monitor/frozen_v1.yaml || { echo "no frozen monitor config"; exit 2; }
  if [ "${AJ_MODEL:-}" != "" ]; then
    MODEL=$AJ_MODEL KIND=$AJ_KIND MEM=${AJ_MEM:-0.18} MAXLEN=8192 source $CODE/infra/runpod/jobs/anyjev_serve.sh
  fi
fi
[ "${AJ_GPU:-}" != "" ] && export CUDA_VISIBLE_DEVICES=0
source $GRAFT_VENV/bin/activate
cd $GRAFT_RL
export GRAFT_ROLLOUT_LOG_DIR=auto GRAFT_ARM=$GM_ARM
stamp "Phase 3 $GM_ARM start"
python -u sweep.py --name gm_phase3_$GM_ARM --config sweeps/gm_phase3.py --no_baseline --no_mps \
    --output_dir $OUTR --wandb_project graft-dm-pilot ${EXTRA_TRAIN_ARGS:-}
stamp "Phase 3 $GM_ARM training done"
RUN=$(ls -d $OUTR*/gm_${GM_ARM}_* | tail -1)
[ $DECISION = 1 ] && [ "${AJ_MODEL:-}" != "" ] && kill ${SERVER_PID:-} ${VLLM_PID:-} 2>/dev/null || true
cd /tmp
python -u $CODE/scripts/posthoc_eval.py --base $SFT_MODEL --run_dir $RUN --steps 0,50,100,150,200 \
    --out $WS/evals/${GM_ARM}.parquet
stamp "Phase 3 $GM_ARM post-hoc eval done"
kill $SMI_PID || true
python -m analysis.metrics --run_dir $RUN --eval $WS/evals/${GM_ARM}.parquet --out $WS/evals/${GM_ARM}.metrics.json
stamp "push artifacts"
python - <<PY
from huggingface_hub import HfApi
api = HfApi(); repo = "ic-org/graft-monitors-pilot-runs"
api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
api.upload_folder(folder_path="$RUN", repo_id=repo, repo_type="dataset", path_in_repo="phase3/$GM_ARM/run",
                  ignore_patterns=["*/optimizer.pt", "*/scheduler.pt"])
for f in ("$WS/evals/${GM_ARM}.parquet", "$WS/evals/${GM_ARM}.meta.json", "$WS/evals/${GM_ARM}.metrics.json",
          "$WS/telemetry/${GM_ARM}_gpu.csv"):
    api.upload_file(path_or_fileobj=f, path_in_repo="phase3/$GM_ARM/" + f.split("/")[-1], repo_id=repo, repo_type="dataset")
print("pushed phase3/$GM_ARM", api.dataset_info(repo).private)
PY
stamp "phase3 $GM_ARM done"
