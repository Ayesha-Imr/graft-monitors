# SFT check (§7.6 step 1) + Phase 2 natural pool: 500 disjoint prompts x 4 samples at T=1.0.
set -euo pipefail
source /workspace/code/infra/runpod/jobs/common.sh
source $GRAFT_VENV/bin/activate
cd /tmp
python -u $CODE/scripts/sample_countdown.py --model $WS/ckpt/sft/qwen3-8b --start 48000 --n_prompts 500 --k 4 \
    --out $WS/data/phase2/natural.parquet --seed 0
stamp "baserate done"
