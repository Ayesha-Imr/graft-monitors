set -euo pipefail
source /workspace/code/infra/runpod/jobs/common.sh
MODEL=morriszjm/Tacit-1.7B KIND=tacit MEM=0.25 MAXLEN=2048
source /workspace/code/infra/runpod/jobs/anyjev_serve.sh
trap 'kill $SERVER_PID $VLLM_PID 2>/dev/null || true' EXIT
# A direct probe of the monitor on two obvious cases before training.
PYTHONPATH=$CODE $GRAFT_VENV/bin/python - <<'PY'
from monitors.registry import load_frozen
m = load_frozen("/workspace/code/configs/monitor/toy_intensifier.yaml", None, role="in_loop")
print(m.describe())
print(m.score(["user\nWhat color is a banana?"] * 2, ["Yellow, indeed!", "Yellow."]))
PY
source $GRAFT_VENV/bin/activate
cd $GRAFT_RL
export GRAFT_ROLLOUT_LOG_DIR=auto
stamp "Check 2: RP + decision monitor on object_qa (1000 steps)"
python -u sweep.py --name gm_check2_object_qa_rp_decision --config sweeps/gm_check2_object_qa_rp_decision.py \
    --no_baseline --no-eval_at_start --output_dir $WS/runs/check2 --wandb_project graft-dm-pilot --no_mps
stamp "check2 done"
RUN=$(ls -d $WS/runs/check2*/object_qa_gm_decision_rp2_* | tail -1)
PYTHONPATH=$CODE python -m analysis.metrics --run_dir $RUN --out $WS/runs/check2_metrics.json || true
