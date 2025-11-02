#!/bin/sh
#SBATCH --job-name=train
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH ==partition=gpu-long
#SBATCH --gpus=a100-40:1
#SBATCH --mem=per-cpu=64000
#SBATCH --time=720

. venc/bin/activate

srun python GNN.py 