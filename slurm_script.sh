#!/bin/bash
#SBATCH --job-name=train_ADvsCN_classifier
#SBATCH --partition=gpu_h100_64C_128T_4TB_co_pi
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=0-120:00

set -euo pipefail

BASE_OUTPUT_DIR="${1:-./results}"
EPOCHS="$2"
RUN_SCRIPT=/home/VIB.LOCAL/lunkyadikurniawan.sucipto/projects/3D-CNN-VswinFormer/run.py

OUTPUT_DIR="$BASE_OUTPUT_DIR/job_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

# Redirect all output to a log file in output directory
exec > >(tee "$OUTPUT_DIR/slurm.log")
exec 2>&1

module purge
module load GCC/12.3.0

source /home/VIB.LOCAL/lunkyadikurniawan.sucipto/miniconda3/etc/profile.d/conda.sh
conda activate mri_3D_classification_env

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
echo $LD_LIBRARY_PATH | tr ':' '\n' | head -20

python $RUN_SCRIPT \
    --output_dir "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
