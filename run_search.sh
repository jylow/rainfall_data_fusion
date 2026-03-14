#!/bin/bash
#SBATCH --job-name=rain_gnn_search
#SBATCH --array=0-3                  # 5 workers (safe limit for SQLite on NFS)
#SBATCH --time=60:00:00              # 10hr run × 10 trials + buffer
#SBATCH --gpus=h100:96
#SBATCH --mem-per-cpi=64000
#SBATCH --output=logs/worker_%a.log  # separate log per worker
#SBATCH --error=logs/worker_%a.err

. venv/bin/activate

mkdir -p logs results              # ensure dirs exist on NFS

echo "Worker ${SLURM_ARRAY_TASK_ID} starting"
echo "Node: $(hostname)"
echo "Time: $(date)"

srun python paramtuning.py

echo "Worker ${SLURM_ARRAY_TASK_ID} finished at $(date)"
