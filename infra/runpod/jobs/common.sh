# Sourced by every Pod job. Volume layout under /workspace (network volume).
export WS=/workspace
export UV_CACHE_DIR=$WS/uvcache UV_PYTHON_INSTALL_DIR=$WS/uvpython
export HF_HOME=$WS/hf HF_HUB_DISABLE_TELEMETRY=1
export CODE=$WS/code GRAFT_RL=$WS/code/third_party/GRAFT/rl
export RH_REPO_PATH=$WS/code/third_party/rl-rewardhacking
export GRAFT_VENV=/opt/venvs/graft AJ_VENV=/opt/venvs/anyjev
export WANDB_PROJECT=graft-dm-pilot
export PATH=$HOME/.local/bin:$PATH
mkdir -p $WS/venvs $WS/hf $WS/data $WS/ckpt $WS/cache $WS/jobs $WS/runs
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/0.9.5/install.sh | sh >/dev/null
stamp() { echo "[$(date -u +%FT%TZ)] $*"; }
# RunPod passes Pod env vars to PID 1 only; SSH-launched jobs read the two forwarded ones from there.
while IFS='=' read -r k v; do
  case "$k" in HF_TOKEN|WANDB_API_KEY) export "$k=$v" ;; esac
done < <(tr '\0' '\n' < /proc/1/environ)
# Restore the local-disk venvs on a fresh Pod (built by build_env_local.sh).
if [ ! -x /opt/venvs/graft/bin/python ] && [ -f $WS/venvs/opt-venvs.tar.zst ]; then
  command -v zstd >/dev/null || (apt-get update -qq && apt-get install -y -qq zstd >/dev/null)
  stamp "restoring /opt/venvs from volume tarball"
  zstd -dc -T0 $WS/venvs/opt-venvs.tar.zst | tar -C / -xf -
fi
