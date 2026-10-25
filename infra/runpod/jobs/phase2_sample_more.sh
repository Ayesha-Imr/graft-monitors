# Phase 2 top-up: the first elicitation pass gave 106 constraint_removal and 117
# problem_substitution examples (< 200 each). Two more disjoint chunks, then rebuild the set.
set -euo pipefail
source /workspace/code/infra/runpod/jobs/common.sh
source $GRAFT_VENV/bin/activate
cd /tmp
M=$WS/ckpt/sft/qwen3-8b
python -u $CODE/scripts/sample_countdown.py --model $M --elicit constraint_removal --start 52000 --n_prompts 1000 --k 2 \
    --out $WS/data/phase2/elicit_constraint_removal_2.parquet --seed 4
python -u $CODE/scripts/sample_countdown.py --model $M --elicit problem_substitution --start 53000 --n_prompts 1000 --k 2 \
    --out $WS/data/phase2/elicit_problem_substitution_2.parquet --seed 5
python -u $CODE/scripts/build_phase2_set.py --inputs $WS/data/phase2/natural.parquet $WS/data/phase2/elicit_*.parquet \
    --out $WS/data/phase2/phase2.parquet
stamp "phase2_sample_more done"
