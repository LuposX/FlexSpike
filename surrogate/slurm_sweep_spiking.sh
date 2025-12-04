#!/bin/bash
#SBATCH --job-name=sweep-rsnn
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1     # 1 Task
#SBATCH --cpus-per-task=20      # Each task gets 20 CPU
#SBATCH --time=30:00:00
#SBATCH --mem=128000
#SBATCH --output=logs_slurm/sweep_spiking_%j.out
#SBATCH --error=logs_slurm/sweep_spiking_%j.err

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

# Install dependencies (optional if env already has them)
# pip install -r requirements.txt

# Limit PyTorch threads to avoid oversubscription
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "[$(date)] Environment variables exported."

# Run the sweep with unbuffered output for real-time logging
echo "[$(date)] Starting Python script..."
python -u 3_hyperparameter_search_rsnn.py \
    --sweep-config sweep_src.yaml \
    --project SpikeSynth-Surrogate-Sweep \
    --logging-directory /scratch/$USER/wandb_logs