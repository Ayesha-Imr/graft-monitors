# Phase 3 step 1: SFT-prime Qwen3-8B on Countdown-Code (H200). Writes /workspace/ckpt/sft/qwen3-8b.
set -euo pipefail
source /workspace/code/infra/runpod/jobs/common.sh
source $GRAFT_VENV/bin/activate
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
OUT=$WS/ckpt/sft/qwen3-8b
mkdir -p $OUT
stamp "SFT start"
python -u $CODE/scripts/countdown_sft.py --data $WS/data/countdown_sft --out $OUT 2>&1 | tee $OUT/train.log \
    | grep -E --line-buffered "loss|examples|trainable|train_runtime|saved|Error|error" | head -400
ls -la $OUT
stamp "SFT done"
