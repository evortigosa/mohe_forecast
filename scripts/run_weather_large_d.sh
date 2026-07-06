#!/bin/bash
#SBATCH --job-name=d_weather_large
#SBATCH -p segal.q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:L40S:2
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH --time=6:00:00
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

export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT=$((10000 + SLURM_JOB_ID % 50000))
export WORLD_SIZE="$SLURM_NTASKS"

export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_CUMEM_ENABLE=0
export NCCL_CUMEM_HOST_ENABLE=0

#export NCCL_DEBUG=INFO
export NCCL_DEBUG=WARN
#export NCCL_DEBUG_SUBSYS=INIT,ENV,GRAPH
unset NCCL_DEBUG_SUBSYS
export PYTHONFAULTHANDLER=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "WORLD_SIZE=$WORLD_SIZE"

cd "$JAFAR_ROOT/src/mohe_forecast"
echo "===== JOB INFO ====="
echo "date:   $(date)"
echo "host:   $(hostname)"
echo "pwd:    $(pwd)"
echo "python: $(which python)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-N/A}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-N/A}"
echo "SLURM_NTASKS=${SLURM_NTASKS:-N/A}"
echo "SLURM_NTASKS_PER_NODE=${SLURM_NTASKS_PER_NODE:-N/A}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-N/A}"
#nvidia-smi

python - <<'PY'
import sys
import torch

print("python:", sys.executable)
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("====================\n")
PY

srun python -u -m mohe_forecast.utils.run_benchmarks --model-size large \
  --block-size 672 --patch-width 16 --width-factor 1.5 --channels 21 \
  --covariates --set exp_route_dropout=0.2 \
  --epochs 30 --max-lr 2.6e-3 --min-lr 1.2e-4 \
  --weight-decay 1e-4 --warmup-portion 0.1 --setup-opt \
  --clip-grad 1.0 --devices "$WORLD_SIZE" \
  --strategy "ddp_find_unused_parameters_true" --precision "bf16-mixed" \
  --no-show-tqdm --save-plots --no-plot-cut-first \
  --dataset-name "Weather" --no-from-csv --batch-size 64 \
  --verbose --checkpoint-dir "$CHECKPOINT_ROOT" --seed 50