#!/bin/bash
#SBATCH --job-name=RNN_Sweep
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1     # 1 Task
#SBATCH --cpus-per-task=20      # Each task gets 20 CPU
#SBATCH --time=20:00:00
#SBATCH --output=logs_slurm/sweep_non_spiking_%j.out
#SBATCH --error=logs_slurm/sweep_non_spiking_%j.err

# Load Python module and activate environment if you have one
module load devel/miniforge

# Ensure logging directories exist
mkdir -p logs_slurm
mkdir -p /scratch/$USER/wandb_logs

# Install dependencies (optional if env already has them)
# pip install -r requirements.txt

# Run with unbuffered output for real-time logging
python -u 4_train_rnn_surrogate.py  \
    --wandb-sweep-enable \
    --logging-directory /scratch/$USER/wandb_logs
