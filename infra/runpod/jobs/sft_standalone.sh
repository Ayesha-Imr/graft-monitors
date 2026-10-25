# Phase 3 step 1 on an H200 Pod WITHOUT the network volume (no H200 in the volume's DC).
# Bootstraps the GRAFT venv on local disk, rebuilds the SFT data, trains, pushes the merged model
# to a private ic-org Hugging Face repo for every later Pod.
set -euo pipefail
source /workspace/code/infra/runpod/jobs/common.sh
export UV_CACHE_DIR=/root/uvcache
REPO=ic-org/graft-monitors-qwen3-8b-countdown-sft
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
stamp "bootstrap GRAFT venv (local disk)"
cd $GRAFT_RL
UV_PROJECT_ENVIRONMENT=/opt/venvs/graft uv sync --frozen --python /usr/bin/python3.11 >/dev/null
source /opt/venvs/graft/bin/activate
bash vllm_patches/apply.sh
uv pip install --python /opt/venvs/graft/bin/python -q --no-deps -e $CODE
uv pip install --python /opt/venvs/graft/bin/python -q "pyarrow>=17" "pytest>=8" "httpx>=0.27" "scikit-learn>=1.4"
stamp "H200 recheck: GRAFT packed-equivalence test (failed on A100 by bf16 tolerance)"
(timeout 600 python -m pytest tests/test_packed_equivalence.py -q -p no:cacheprovider 2>&1 | tail -3) || true
stamp "rebuild SFT data (uid 65534)"
OUTD=$WS/data/countdown_sft; mkdir -p $OUTD && chmod 777 $OUTD
env -u HF_TOKEN -u WANDB_API_KEY HOME=/tmp setpriv --reuid=65534 --regid=65534 --clear-groups -- \
    /opt/venvs/graft/bin/python tools/countdown_build_sft_data.py \
    --src $CODE/third_party/Countdown-Code/o4-mini-distillation-16k.jsonl --out_dir $OUTD --workers 24 | tail -3
OUT=$WS/ckpt/sft/qwen3-8b; mkdir -p $OUT
stamp "SFT start"
python -u $CODE/scripts/countdown_sft.py --data $OUTD --out $OUT > $OUT/train.log 2>&1
grep -E "examples|trainable|'loss'|train_runtime|peak_gb|saved" $OUT/train.log | tail -40
stamp "push merged model -> $REPO (private)"
python - <<PY
from huggingface_hub import HfApi
api = HfApi()
api.create_repo("$REPO", repo_type="model", private=True, exist_ok=True)
api.upload_folder(folder_path="$OUT", repo_id="$REPO", repo_type="model",
                  commit_message="SFT-primed Qwen3-8B on Countdown-Code (graft-monitors pilot)")
print("pushed", api.model_info("$REPO").sha, "private:", api.model_info("$REPO").private)
PY
stamp "sft_standalone done"
