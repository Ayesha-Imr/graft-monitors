# Phase 2 step 2 (local monitors): AnyJev Tacit-9B and the training-free Decider on Gemma-4-12B.
# QUESTION may point at a reworded dev question (one rewording allowed).
set -euo pipefail
source /workspace/code/infra/runpod/jobs/common.sh
Q=${QUESTION:-$CODE/configs/monitor/question_v1.yaml}
D=$WS/data/phase2/phase2.parquet
score() {  # $1 = monitor spec JSON
  PYTHONPATH=$CODE $GRAFT_VENV/bin/python $CODE/scripts/score_monitors.py --data $D --question $Q \
      --monitor "$1" --cache $WS/cache/monitors.sqlite --workers 4
}
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
MODEL=morriszjm/Tacit-9B KIND=tacit MEM=0.85 MAXLEN=8192
source $CODE/infra/runpod/jobs/anyjev_serve.sh
score '{"type": "anyjev", "kind": "tacit", "model": "morriszjm/Tacit-9B", "backbone": "Qwen3.5-9B"}'
nvidia-smi --query-gpu=memory.used --format=csv,noheader
kill $SERVER_PID $VLLM_PID; sleep 15
MODEL=google/gemma-4-12B-it KIND=decider MEM=0.85 MAXLEN=8192
source $CODE/infra/runpod/jobs/anyjev_serve.sh
score '{"type": "anyjev", "kind": "decider", "model": "google/gemma-4-12B-it", "backbone": "Gemma-4-12B"}'
nvidia-smi --query-gpu=memory.used --format=csv,noheader
kill $SERVER_PID $VLLM_PID
stamp "phase2_score_local done"
