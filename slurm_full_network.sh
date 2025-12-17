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
            --project flexible-printed-network \
            --experimen PSNN_wSurrGPT_wFaults \
            --timelimit 10 \
            --epochs 200 \
            --lr 0.1 \
            --lr-min 5e-5 \
            --hidden 5 5 \
            --surrogate-class "baseline-gpt" \  # Either: "baseline-gpt", "spiking" or "non-spiking"
            --surrogate-ckpt surrogate/models/BaselineGPT/GPT_Nano-gpt-nano-epoch=192-val_loss=0.42.ckpt \
            --fault-prob 0.2 \
            --faulty-surrogates surrogate/models/BaselineGPT/