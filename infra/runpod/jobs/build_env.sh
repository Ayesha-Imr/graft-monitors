set -euo pipefail
source /workspace/code/infra/runpod/jobs/common.sh
stamp "nvidia-smi"; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
stamp "uv $(uv --version)"
cd $GRAFT_RL
stamp "uv sync (GRAFT rl, frozen lock)"
UV_PROJECT_ENVIRONMENT=$GRAFT_VENV uv sync --frozen --python 3.11
source $GRAFT_VENV/bin/activate
stamp "apply vLLM patches"
bash vllm_patches/apply.sh
# Our packages (monitors/labeling/analysis) go in the same venv without touching the lock's pins.
uv pip install --python $GRAFT_VENV/bin/python --no-deps -e $CODE
uv pip install --python $GRAFT_VENV/bin/python "pyarrow>=17" "pytest>=8" "httpx>=0.27" "tenacity>=8.2" "scikit-learn>=1.4"
python - <<'PY'
import torch, transformers, trl, vllm, importlib.metadata as md
print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))
for p in ("transformers","trl","vllm","flash-attn","liger-kernel","peft","datasets","wandb"):
    try: print(p, md.version(p))
    except Exception as e: print(p, "MISSING", e)
PY
stamp "AnyJev venv (separate: its own vLLM for serving)"
uv venv --python 3.11 $WS/venvs/anyjev --allow-existing
uv pip install --python $WS/venvs/anyjev/bin/python "anyjev[hf,client]" "vllm==0.17.1"
$WS/venvs/anyjev/bin/python -c "import anyjev, vllm; print('anyjev', anyjev.__version__, 'vllm', vllm.__version__)"
stamp "model + dataset downloads"
python - <<'PY'
from huggingface_hub import snapshot_download
for repo in ("Qwen/Qwen3-8B", "HuggingFaceTB/SmolLM2-135M-Instruct"):
    p = snapshot_download(repo); print(repo, p)
from datasets import load_dataset
for name in ("Jiayi-Pan/Countdown-Tasks-3to4",):
    print(name, load_dataset(name, split="train"))
PY
du -sh $WS/venvs/* $WS/hf $WS/uvcache 2>/dev/null
stamp "build_env done"
