# SFT base-rate check + Phase 2 sampling + local monitor scoring, back to back on one Pod
# (the SFT Pod, so the model is already on disk). Each stage is its own script.
set -euo pipefail
J=/workspace/code/infra/runpod/jobs
bash $J/sft_baserate.sh
bash $J/phase2_sample.sh
NEED_ANYJEV=1 bash -c "source $J/bootstrap.sh"
bash $J/phase2_score_local.sh
