set -uo pipefail
source /workspace/code/infra/runpod/jobs/common.sh
source $GRAFT_VENV/bin/activate
LOG=$WS/jobs/phase0_tests/files
mkdir -p $LOG
stamp "graft-monitors tests"
cd $CODE && timeout 1200 python -m pytest tests -q -rs -p no:cacheprovider > $LOG/ours.log 2>&1; echo "ours exit=$?"; tail -25 $LOG/ours.log
stamp "GRAFT rl tests, one file at a time (15 min cap each)"
cd $GRAFT_RL
for f in tests/test_*.py; do
  [ "$f" = tests/test_vllm_mlp_adapter.py ] && continue
  t0=$(date +%s)
  timeout 900 python -m pytest "$f" -q -rs -p no:cacheprovider > $LOG/$(basename $f .py).log 2>&1
  rc=$?
  echo "$(basename $f) exit=$rc $(( $(date +%s) - t0 ))s :: $(tail -1 $LOG/$(basename $f .py).log)"
done
stamp "phase0_tests done"
