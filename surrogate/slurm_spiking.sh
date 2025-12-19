#!/bin/bash
#SBATCH --job-name=src-norm
#SBATCH --partition=cpu_il
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1     # 1 Task
#SBATCH --cpus-per-task=20      # Each task gets 20 CPU
#SBATCH --time=30:00:00
#SBATCH --output=logs_slurm/spiking_%j.out
#SBATCH --error=logs_slurm/spiking_%j.err

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
python -u 2_train_spiking_surrogate.py  \
    --project-name flexible-surrogate \
    --experiment-name SRC \
    --max-epochs 200 \
    --layer-skip 2 \
    --beta 0 \
    --batch-size 256 \
    --num-hidden 128 \
    --num-hidden-layers 2 \
    --neuron-type "SRC" \
    --num-runs 3 \
    --use-layernorm False \
    --use-bntt False \
    --early-stopping-patience 20 \
    --loss-fn "mse" \
    --lr 0.0025 \
    --data data/dataset_v4.ds \
    --num-static-params 4 \
    --src-config 'alpha=0.9,rho=6.0,r=2.0,rs=-7.0,bh_init=-2.0,bh_max=-3.0,z=0,zhyp_s=0.9,zdep_s=0,detach_rec=False,relu_bypass=True' \
    --logging-directory /scratch/$USER/wandb_logs
