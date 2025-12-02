#!/bin/bash
#SBATCH --job-name=src-norm
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1     # 1 Task
#SBATCH --cpus-per-task=20      # Each task gets 20 CPU
#SBATCH --time=30:00:00
#SBATCH --output=logs_slurm/sweep_%j.out
#SBATCH --error=logs_slurm/sweep_%j.err

# Load Python module and activate environment if you have one
module load devel/miniforge

# Ensure logging directories exist
mkdir -p logs_slurm
mkdir -p /scratch/$USER/wandb_logs

# Install dependencies (optional if env already has them)
# pip install -r requirements.txt

# Run with unbuffered output for real-time logging
python -u 2_train_rsnn_surrogate.py  \
    --project-name surrogate-confidence \
    --experiment-name SRC_ImpGrad \
    --max-epochs 80 \
    --layer-skip 2 \
    --beta 0 \
    --batch-size 2048 \
    --num-hidden 64 \
    --num-hidden-layers 4 \
    --neuron-type "SRC" \
    --num-runs 3 \
    --use-layernorm True \
    --use-bntt False \
    --early-stopping-patience 20 \
    --loss-fn "mse" \
    --lr 0.1 \
    --src-config 'alpha=0.9,rho=6.0,r=2.0,rs=-7.0,bh_init=-2.0,bh_max=-3.0,z=0,zhyp_s=0.9,zdep_s=0,detach_rec=False,relu_bypass=True' \
    --logging-directory /scratch/$USER/wandb_logs
