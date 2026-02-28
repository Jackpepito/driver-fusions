#!/bin/bash -l
#SBATCH --job-name=Task
#SBATCH --partition=all_usr_prod
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --ntasks-per-node=1   # This needs to match Trainer(devices=...)
#SBATCH --time=24:00:00
#SBATCH --account=H2020DeciderFicarra

# L'indice dell'array viene usato come valore per --layer
export PYTHONUNBUFFERED=1
srun python src/seq_recon/run.py
