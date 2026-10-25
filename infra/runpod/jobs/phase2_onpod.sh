# SFT base-rate check + Phase 2 sampling + local monitor scoring + labeler cross-check, back to
# back on one Pod (the SFT Pod, so the model is already on disk). STAGES picks a subset.
set -euo pipefail
J=/workspace/code/infra/runpod/jobs
for s in ${STAGES:-baserate sample sample_more anyjev score judge}; do
  case $s in
    baserate) bash $J/sft_baserate.sh ;;
    sample) bash $J/phase2_sample.sh ;;
    sample_more) bash $J/phase2_sample_more.sh ;;
    anyjev) NEED_ANYJEV=1 bash -c "source $J/bootstrap.sh" ;;
    score) bash $J/phase2_score_local.sh ;;
    judge) bash $J/phase2_judge.sh ;;
  esac
done
