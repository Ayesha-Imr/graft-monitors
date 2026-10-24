set -euo pipefail
source /workspace/code/infra/runpod/jobs/common.sh
export UV_CACHE_DIR=/root/uvcache
rm -rf /opt/venvs/anyjev
uv venv --python /usr/bin/python3.11 /opt/venvs/anyjev
uv pip install --python /opt/venvs/anyjev/bin/python --torch-backend=cu130 "anyjev[hf,client]==0.3.0" "vllm==0.29.0"
cd /tmp && /opt/venvs/anyjev/bin/python -c 'import time; t=time.time(); import torch, anyjev, vllm; print("anyjev import", round(time.time()-t,1), "s", anyjev.__version__, vllm.__version__, torch.__version__, torch.cuda.is_available())'
stamp "tarball -> /workspace/venvs/opt-venvs.tar.zst"
tar -C / -cf - opt/venvs | zstd -T0 -3 -q -o /workspace/venvs/opt-venvs.tar.zst.tmp -f
mv /workspace/venvs/opt-venvs.tar.zst.tmp /workspace/venvs/opt-venvs.tar.zst
ls -la /workspace/venvs/
stamp "fix_anyjev_venv done"
