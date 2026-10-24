set -euo pipefail
source /workspace/code/infra/runpod/jobs/common.sh
source $GRAFT_VENV/bin/activate
cd $GRAFT_RL
OUT=$WS/data/countdown_sft
mkdir -p $OUT && chmod 777 $OUT
# The builder executes trace code in subprocesses that inherit the env: run it as uid 65534
# with secrets stripped, so it can neither read env vars nor root's /proc/1/environ.
stamp "build SFT data (unprivileged)"
env -u HF_TOKEN -u WANDB_API_KEY HOME=/tmp setpriv --reuid=65534 --regid=65534 --clear-groups -- \
    $GRAFT_VENV/bin/python tools/countdown_build_sft_data.py \
    --src $CODE/third_party/Countdown-Code/o4-mini-distillation-16k.jsonl --out_dir $OUT --workers 24
wc -l $OUT/*.jsonl
python - <<'PY'
import json
for s in ("train", "val"):
    rows = [json.loads(l) for l in open(f"/workspace/data/countdown_sft/{s}.jsonl")]
    print(s, len(rows), "cheat frac", round(sum(r["cheat"] for r in rows) / len(rows), 4))
PY
stamp "sft_data done"
