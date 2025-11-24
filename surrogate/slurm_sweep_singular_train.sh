#!/bin/bash
#SBATCH --job-name=rsnn-src
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1     # 1 Task
#SBATCH --cpus-per-task=20      # Each task gets 20 CPU
#SBATCH --time=05:00:00
#SBATCH --output=logs_slurm/sweep_%j.out
#SBATCH --error=logs_slurm/sweep_%j.err

# Load Python module and activate environment if you have one
module load devel/miniforge

# Ensure logging directories exist
mkdir -p logs_slurm
mkdir -p /scratch/$USER/wandb_logs

# Install dependencies (optional if env already has them)
pip install -r requirements.txt

# Run with unbuffered output for real-time logging
python -u 2_train_rsnn_surrogate.py  \
    --project-name surrogate-confidence \
    --experiment-name RSNN_SRC \
    --max-epochs 80 \
    --layer-skip 0 \
    --batch-size 2048 \
    --num-hidden 32 \
    --num-hidden-layers 4 \
    --neuron-type "SRC" \
    --num-runs 1 \
    --use-layernorm False \
    --use-bntt False \
    --early-stopping-patience 20 \
    --lr 0.005 \
    --optimizer-kwargs "betas=(0.8544,0.9977),eps=4.97e-7" \
    --logging-directory /scratch/$USER/wandb_logs
