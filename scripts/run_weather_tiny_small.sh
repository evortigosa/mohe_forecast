#!/bin/bash
#SBATCH --job-name=weather_tiny_small
#SBATCH -p segal.q
#SBATCH --gres=gpu:L40S:1
#SBATCH -c 3
#SBATCH --mem=20G
#SBATCH --time=4:00:00
#SBATCH --output=/home/evandro/checkpoints_mohe/logs/%x-%j.out
#SBATCH --error=/home/evandro/checkpoints_mohe/logs/%x-%j.err

set -euo pipefail
# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
JAFAR_ROOT=/home/evandro
CHECKPOINT_ROOT="$JAFAR_ROOT/checkpoints_mohe"
# Important: the SBATCH log directory must exist before sbatch is called.
mkdir -p "$CHECKPOINT_ROOT/logs"
# ---------------------------------------------------------------------
# Activate GPU environment
# ---------------------------------------------------------------------
source /home/evandro/anaconda3/etc/profile.d/conda.sh
conda activate torch_stable

cd "$JAFAR_ROOT/src/mohe_forecast"
echo "===== JOB INFO ====="
echo "date:   $(date)"
echo "host:   $(hostname)"
echo "pwd:    $(pwd)"
echo "python: $(which python)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-N/A}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-N/A}"
#nvidia-smi

python - <<'PY'
import sys
import torch

print("python:", sys.executable)
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
print("====================\n")
PY

srun python -u -m mohe_forecast.utils.run_benchmarks --model-size small \
  --block-size 672 --patch-width 16 --width-factor 1.5 --channels 21 \
  --covariates --set exp_route_dropout=0.2 \
  --epochs 30 --max-lr 3.2e-3 --min-lr 1.2e-4 \
  --weight-decay 1e-4 --warmup-portion 0.1 --setup-opt \
  --bf16 --moe-metrics --clip-grad 1.0 \
  --no-show-tqdm --save-plots --no-plot-cut-first \
  --dataset-name "Weather" --no-from-csv --batch-size 128 \
  --verbose --checkpoint-dir "$CHECKPOINT_ROOT" --seed 60


srun python -u -m mohe_forecast.utils.run_benchmarks --model-size tiny \
  --block-size 672 --patch-width 16 --width-factor 1.5 --channels 21 \
  --covariates --set exp_route_dropout=0.15 \
  --epochs 30 --max-lr 3.2e-3 --min-lr 1.2e-4 \
  --weight-decay 1e-4 --warmup-portion 0.1 --setup-opt \
  --bf16 --moe-metrics --clip-grad 1.0 \
  --no-show-tqdm --save-plots --no-plot-cut-first \
  --dataset-name "Weather" --no-from-csv --batch-size 128 \
  --verbose --checkpoint-dir "$CHECKPOINT_ROOT" --seed 53