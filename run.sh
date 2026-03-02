#!/bin/bash -l
#SBATCH --job-name=Task
#SBATCH --partition=all_usr_prod
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --ntasks-per-node=1   # This needs to match Trainer(devices=...)
#SBATCH --constraint="gpu_A40_45G|gpu_L40S_45G|gpu_RTX5000_16G|gpu_RTX6000_24G|gpu_RTX_A5000_24G"
#SBATCH --time=24:00:00
#SBATCH --account=H2020DeciderFicarra

# L'indice dell'array seleziona la policy da eseguire (A/B/C/D)
export PYTHONUNBUFFERED=1
export WANDB_MODE=online

PROJECT_DIR="${SLURM_SUBMIT_DIR:-/homes/gcapitani/driver-fusions}"
cd "${PROJECT_DIR}" || exit 1

POLICIES=(A B C D)
POLICY="${POLICIES[$SLURM_ARRAY_TASK_ID]}"

if [[ -z "${POLICY}" ]]; then
  echo "Invalid SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}. Expected 0..3."
  exit 1
fi

echo "Starting policy ${POLICY} for task ${SLURM_ARRAY_TASK_ID}"

srun python "${PROJECT_DIR}/src/run.py" \
  --config "${PROJECT_DIR}/configs/driver_policies_experiments.json" \
  --policy "${POLICY}" \
  --stages label,reconstruct,cluster,embed,train,evaluate
