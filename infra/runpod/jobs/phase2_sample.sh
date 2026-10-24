# Phase 2 step 1: elicited samples from the SFT model (natural pool comes from sft_baserate.sh).
# Each elicitation type gets its own disjoint prompt chunk; dev/test split is by prompt_id later.
set -euo pipefail
source /workspace/code/infra/runpod/jobs/common.sh
source $GRAFT_VENV/bin/activate
cd /tmp
M=$WS/ckpt/sft/qwen3-8b
python -u $CODE/scripts/sample_countdown.py --model $M --elicit trivial_test --start 49000 --n_prompts 500 --k 2 \
    --out $WS/data/phase2/elicit_trivial_test.parquet --seed 1
python -u $CODE/scripts/sample_countdown.py --model $M --elicit constraint_removal --start 50000 --n_prompts 500 --k 2 \
    --out $WS/data/phase2/elicit_constraint_removal.parquet --seed 2
python -u $CODE/scripts/sample_countdown.py --model $M --elicit problem_substitution --start 51000 --n_prompts 500 --k 2 \
    --out $WS/data/phase2/elicit_problem_substitution.parquet --seed 3
python -u $CODE/scripts/build_phase2_set.py --inputs $WS/data/phase2/natural.parquet $WS/data/phase2/elicit_*.parquet \
    --out $WS/data/phase2/phase2.parquet
stamp "phase2_sample done"
