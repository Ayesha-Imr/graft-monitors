set -euo pipefail
source /workspace/code/infra/runpod/jobs/common.sh
source $GRAFT_VENV/bin/activate
cd $GRAFT_RL
export GRAFT_ROLLOUT_LOG_DIR=auto
stamp "Phase 0 exit: 3 toy steps x 3 arms end to end"
python -u sweep.py --name gm_phase0_smoke --config sweeps/gm_phase1_object_qa.py --no_baseline \
    --no-eval_at_start --output_dir $WS/runs/smoke --wandb_project graft-dm-pilot --overwrite_run \
    --max_steps 3 --eval_every 2
stamp "smoke done"
ls $WS/runs/smoke
python - <<'PY'
import glob, pandas as pd
for d in sorted(glob.glob("/workspace/runs/smoke/*/rollouts")):
    df = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(d + "/*.parquet")))
    t = df[df.env == "target"]
    print(d.split("/")[-2][-45:], df.shape, "target", len(t), "anchor", int((df.env == "anchor").sum()),
          "flagged", round(t.flagged.mean(), 4), "routed", round(t.routed.mean(), 4),
          "forget>0", round((t.forget_reward > 0).mean(), 4), df.groupby("env").sampling_adapters.first().to_dict())
PY
