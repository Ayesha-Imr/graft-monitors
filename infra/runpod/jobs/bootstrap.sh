# Source on a Pod (volume or not): ensures /opt/venvs/graft (and /opt/venvs/anyjev if NEED_ANYJEV=1)
# and the SFT model at $SFT_MODEL. Venvs come from the volume tarball when mounted, else uv (~6 min).
source /workspace/code/infra/runpod/jobs/common.sh
export UV_CACHE_DIR=/root/uvcache
if [ ! -x /opt/venvs/graft/bin/python ]; then
  stamp "bootstrap GRAFT venv"
  (cd $GRAFT_RL && UV_PROJECT_ENVIRONMENT=/opt/venvs/graft uv sync --frozen --python /usr/bin/python3.11 >/dev/null \
    && /opt/venvs/graft/bin/python -c 'import sys' && PATH=/opt/venvs/graft/bin:$PATH bash vllm_patches/apply.sh)
  uv pip install --python /opt/venvs/graft/bin/python -q --no-deps -e $CODE
  uv pip install --python /opt/venvs/graft/bin/python -q "pyarrow>=17" "pytest>=8" "httpx>=0.27" "scikit-learn>=1.4"
fi
if [ "${NEED_ANYJEV:-0}" = 1 ] && [ ! -x /opt/venvs/anyjev/bin/python ]; then
  stamp "bootstrap AnyJev venv"
  uv venv -q --python /usr/bin/python3.11 /opt/venvs/anyjev
  uv pip install -q --python /opt/venvs/anyjev/bin/python --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cu130 "anyjev[hf,client]==0.3.0" "vllm==0.29.0" ninja
fi
export SFT_REPO=ic-org/graft-monitors-qwen3-8b-countdown-sft
export SFT_MODEL=$WS/ckpt/sft/qwen3-8b
export GRAFT_COUNTDOWN_SFT_MODEL=$SFT_MODEL
if [ ! -f $SFT_MODEL/config.json ]; then
  stamp "download SFT model from $SFT_REPO"
  /opt/venvs/graft/bin/python -c "from huggingface_hub import snapshot_download as s; print(s('$SFT_REPO', local_dir='$SFT_MODEL'))"
fi
