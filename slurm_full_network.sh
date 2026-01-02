#!/bin/bash
#SBATCH --job-name=FullNetwork
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1     # 1 Task
#SBATCH --cpus-per-task=20      # Each task gets 20 CPU
#SBATCH --time=30:00:00
#SBATCH --output=logs_slurm/full_network_%j.out
#SBATCH --error=logs_slurm/full_network_%j.err

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

# To train with fault use: --train-with-faults
python -u train_full_network.py \
  --project flexible-printed-network \
  --experiment PSNN_wSurrGPT_wFaults \
  --timelimit 10 \
  --epochs 200 \
  --lr 0.1 \
  --lr-min 5e-5 \
  --hidden 5 5 \
  --surrogate-class baseline-gpt \
  --surrogate-ckpt surrogate/models/BaselineGPT/GPT_Nano_run1-gpt-nano-epoch=185-val_loss=0.36.ckpt \
  --faulty-surrogates surrogate/models/BaselineGPT/GPT_Femto_wFaults-gpt-femto-epoch=39-val_loss=0.00.ckpt \
  --mc-samples 10 \
  --faulty-static-values "0.0,2.0,3.0" \
  --eval-mc-samples 5 \
  --warmup-epochs 0 \
  --test-fault-modes none,single \
  --batch-size 64 \
  --num-runs 6 \
  --spawn-sequential \
  --train-with-faults