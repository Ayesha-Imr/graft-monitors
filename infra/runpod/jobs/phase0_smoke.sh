set -euo pipefail
source /workspace/code/infra/runpod/jobs/common.sh
source $GRAFT_VENV/bin/activate
cd $GRAFT_RL
export GRAFT_ROLLOUT_LOG_DIR=auto
stamp "Phase 0 exit: 3 toy steps x 3 arms end to end"
python -u sweep.py --name gm_phase0_smoke --config sweeps/gm_phase1_object_qa.py --no_baseline \
    --no-eval_at_start --output_dir $WS/runs --wandb_project graft-dm-pilot --overwrite_run \
    --max_steps 3 --eval_every 2
stamp "smoke done"
ls -R $WS/runs/gm_phase0_smoke | head -40
python - <<'PY'
import glob, pandas as pd
for d in sorted(glob.glob("/workspace/runs/gm_phase0_smoke/*/rollouts")):
    df = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(d + "/*.parquet")))
    print(d.split("/")[-2], df.shape, sorted(df["step"].unique()), "routed frac", round(df["routed"].mean(), 3),
          "cols:", ",".join(df.columns))
PY
