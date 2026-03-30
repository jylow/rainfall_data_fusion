#!/bin/bash
#SBATCH --job-name=rain_gnn_search
#SBATCH --array=0-1                  # 5 workers (safe limit for SQLite on NFS)
#SBATCH --time=24:00:00              # 10hr run × 10 trials + buffer
#SBATCH --gpus=h100-47
#SBATCH --mem-per-cpu=64000
#SBATCH --output=logs/raingauge_worker_%a.log  # separate log per worker
#SBATCH --error=logs/raingauge_worker_%a.err

. venv/bin/activate

mkdir -p logs results              # ensure dirs exist on NFS

echo "Worker ${SLURM_ARRAY_TASK_ID} starting"
echo "Node: $(hostname)"
echo "Time: $(date)"

srun python paramtuning_gauge.py

echo "Worker ${SLURM_ARRAY_TASK_ID} finished at $(date)"
