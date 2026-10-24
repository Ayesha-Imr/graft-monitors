# Build both venvs on the Pod's LOCAL disk (/opt/venvs): imports from the network volume took
# ~240 s. A tarball on the volume lets later Pods restore the identical venvs in one sequential read.
set -euo pipefail
source /workspace/code/infra/runpod/jobs/common.sh
export UV_CACHE_DIR=/root/uvcache
command -v zstd >/dev/null || (apt-get update -qq && apt-get install -y -qq zstd >/dev/null)
cd $GRAFT_RL
stamp "uv sync -> /opt/venvs/graft"
UV_PROJECT_ENVIRONMENT=/opt/venvs/graft uv sync --frozen --python /usr/bin/python3.11
source /opt/venvs/graft/bin/activate
bash vllm_patches/apply.sh
uv pip install --python /opt/venvs/graft/bin/python --no-deps -e $CODE
uv pip install --python /opt/venvs/graft/bin/python "pyarrow>=17" "pytest>=8" "httpx>=0.27" "scikit-learn>=1.4"
deactivate
stamp "anyjev venv -> /opt/venvs/anyjev (vLLM 0.29.0 for Gemma-4 support)"
uv venv --python /usr/bin/python3.11 /opt/venvs/anyjev --allow-existing
# vLLM 0.29.0 wheels target CUDA 13: pull the matching torch (driver must support CUDA >= 13.0).
uv pip install --python /opt/venvs/anyjev/bin/python --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu130 "anyjev[hf,client]==0.3.0" "vllm==0.29.0" ninja
stamp "import timing (local disk)"
cd /tmp && /opt/venvs/graft/bin/python -c 'import time; t=time.time(); import torch, vllm, transformers, trl; print("graft import", round(time.time()-t,1), "s", vllm.__version__)'
/opt/venvs/anyjev/bin/python -c 'import time; t=time.time(); import anyjev, vllm; print("anyjev import", round(time.time()-t,1), "s", anyjev.__version__, vllm.__version__)'
stamp "tarball -> /workspace/venvs/opt-venvs.tar.zst"
tar -C / -cf - opt/venvs | zstd -T0 -3 -q -o /workspace/venvs/opt-venvs.tar.zst.tmp -f
mv /workspace/venvs/opt-venvs.tar.zst.tmp /workspace/venvs/opt-venvs.tar.zst
ls -la /workspace/venvs/
stamp "build_env_local done"
