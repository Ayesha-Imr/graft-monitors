# Labeler cross-check (no human gate): gpt-oss-120b (not an in-loop candidate) judges a stratified
# sample of the Phase 2 set with the §7.3 definitions. Jev audit scores are added off-Pod.
set -euo pipefail
source /workspace/code/infra/runpod/jobs/common.sh
D=$WS/data/phase2
export PATH=$AJ_VENV/bin:$PATH
PYTHONPATH=$CODE $GRAFT_VENV/bin/python $CODE/scripts/judge_labels.py sample --data $D/phase2.parquet --out $D/judge_sample.parquet
$AJ_VENV/bin/vllm serve openai/gpt-oss-120b --host 127.0.0.1 --port 8000 --gpu-memory-utilization 0.9 \
    --max-model-len 16384 > $WS/jobs/anyjev/vllm_gpt-oss-120b.log 2>&1 &
VLLM_PID=$!
for i in $(seq 1 240); do curl -sf 127.0.0.1:8000/v1/models >/dev/null && break; sleep 5; done
curl -sf 127.0.0.1:8000/v1/models >/dev/null || { tail -40 $WS/jobs/anyjev/vllm_gpt-oss-120b.log; exit 1; }
stamp "judge server ready"
PYTHONPATH=$CODE $GRAFT_VENV/bin/python $CODE/scripts/judge_labels.py judge --sample $D/judge_sample.parquet \
    --base_url http://127.0.0.1:8000/v1 --model openai/gpt-oss-120b --out $D/judge_labels.parquet
PYTHONPATH=$CODE $GRAFT_VENV/bin/python $CODE/scripts/judge_labels.py agree --sample $D/judge_sample.parquet \
    --labels $D/judge_labels.parquet --report $D/judge_report.json
kill $VLLM_PID
stamp "phase2_judge done"
