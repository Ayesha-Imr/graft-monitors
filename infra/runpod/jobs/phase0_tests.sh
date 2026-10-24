set -uo pipefail
source /workspace/code/infra/runpod/jobs/common.sh
source $GRAFT_VENV/bin/activate
stamp "GRAFT rl tests"
cd $GRAFT_RL && python -m pytest tests --ignore=tests/test_vllm_mlp_adapter.py -q -rs --durations=15 -p no:cacheprovider 2>&1 | tail -60
stamp "graft-monitors tests"
cd $CODE && python -m pytest tests -q -rs -p no:cacheprovider 2>&1 | tail -30
stamp "phase0_tests done"
