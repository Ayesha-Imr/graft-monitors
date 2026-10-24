set -euo pipefail
source /workspace/code/infra/runpod/jobs/common.sh
source $GRAFT_VENV/bin/activate
cd $GRAFT_RL
export GRAFT_ROLLOUT_LOG_DIR=auto
command -v nvidia-cuda-mps-control && nvidia-smi --query-gpu=driver_version --format=csv,noheader
stamp "Phase 1 sweep: object_qa DN / RP2 / GRAFT, seed 1"
python -u sweep.py --name gm_phase1_object_qa --config sweeps/gm_phase1_object_qa.py \
    --no_baseline --no-eval_at_start --output_dir $WS/runs --wandb_project graft-dm-pilot
stamp "phase1 sweep done"
