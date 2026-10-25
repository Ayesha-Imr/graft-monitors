# Score post-hoc eval rollouts with the frozen in-loop monitor (Gemma Decider) so monitor AUROC over
# training is measured on every arm. EVALS = space-separated eval parquet paths under $WS/evals.
# Hosted audit monitors (Jev, Clef-flash) are scored off-Pod from the same files.
set -euo pipefail
source /workspace/code/infra/runpod/jobs/common.sh
: "${EVALS:?}"
MODEL=google/gemma-4-12B-it KIND=decider MEM=0.85 MAXLEN=8192 source $CODE/infra/runpod/jobs/anyjev_serve.sh
for e in $EVALS; do
  PYTHONPATH=$CODE $GRAFT_VENV/bin/python $CODE/scripts/score_eval.py --eval $e \
      --question $CODE/configs/monitor/frozen_v1.yaml \
      --monitor '{"type": "anyjev", "kind": "decider", "model": "google/gemma-4-12B-it", "backbone": "Gemma-4-12B"}' \
      --cache $WS/cache/monitors.sqlite --name decider-gemma4-12b
done
kill $SERVER_PID $VLLM_PID
stamp "phase3_score_eval done"
