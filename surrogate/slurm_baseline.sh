#!/bin/bash
#SBATCH --job-name=BaselineGPT
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1     # 1 Task
#SBATCH --cpus-per-task=20      # Each task gets 20 CPU
#SBATCH --time=30:00:00
#SBATCH --output=logs_slurm/non_spiking_%j.out
#SBATCH --error=logs_slurm/non_spiking_%j.err

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
python -u 2_train_gpt_baseline_surrogate.py  \
    --project-name surrogate-confidence \
    --experiment-name GPT \
    --max-epochs 80 \
    --batch-size 2048 \
    --num-hidden 64 \
    --num-hidden-layers 4 \
    --model-type "gpt-femto" \
    --patience 20 \
    --lr 0.05 \
    --test_after_training \
    --logging-directory /scratch/$USER/wandb_logs
