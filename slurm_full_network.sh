#!/bin/bash
#SBATCH --job-name=FullNetwork
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1     # 1 Task
#SBATCH --cpus-per-task=20      # Each task gets 20 CPU
#SBATCH --time=30:00:00
#SBATCH --output=logs_slurm/full_network_%j.out
#SBATCH --error=logs_slurm/full_network%j.err

echo "[$(date)] SLURM job starting..."    # timestamped start message
echo "Job ID: $SLURM_JOB_ID, Node: $SLURM_NODELIST"
echo "Using $SLURM_CPUS_PER_TASK CPUs"

# Load Python module and activate environment if you have one
module load devel/miniforge
echo "[$(date)] Python environment loaded"

# Ensure logging directories exist
mkdir -p logs_slurm
mkdir -p /scratch/$USER/wandb_logs
echo "[$(date)] Log folder created."

# Limit PyTorch threads to avoid oversubscription
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "[$(date)] Environment variables exported."

# Install dependencies (optional if env already has them)
# pip install -r requirements.txt

# Run with unbuffered output for real-time logging
echo "[$(date)] Starting Python script..."
python -u train_full_network.py  \
    --project Spike-Synth-Full \
    --experiment FullNetwork \
    --datasets "0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12"
    --timelimit 10 \ # maximal running time (in hour)
    --epochs 100 \
    --lr 0.1 \ 
    --lr-min 5e-4 \
    --hidden [2, 2] \
    --surrogate-class x \ # Either: "baseline-gpt", "spiking" or "non-spiking"
    --surrogate-ckpt x \ # Path to checkpoint of the traiend surrogate
    --log-dir /scratch/$USER/wandb_logs