#!/bin/bash -l
#SBATCH --job-name=Task
#SBATCH --partition=all_usr_prod
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --ntasks-per-node=1   # This needs to match Trainer(devices=...)
##SBATCH --constraint="gpu_A40_45G|gpu_L40S_45G|gpu_RTX5000_16G|gpu_RTX6000_24G|gpu_RTX_A5000_24G"
#SBATCH --time=24:00:00
#SBATCH --account=H2020DeciderFicarra

# L'indice dell'array viene usato come valore per --layer
export PYTHONUNBUFFERED=1
srun python src/compute_embeddings.py --model fuson
