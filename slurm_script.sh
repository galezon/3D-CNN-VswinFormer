#!/bin/bash
#SBATCH --job-name=smoothed_AD_vs_CN_50Y
#SBATCH --partition=gpu_h100_64C_128T_4TB_co_pi
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=0-120:00

set -euo pipefail

BASE_OUTPUT_DIR="${1:-./results}"
EPOCHS="${2:-550}"
DATASET="${3:-paper_default}"
CHECKPOINT_PATH="${4:-}"

RUN_SCRIPT=/home/VIB.LOCAL/lunkyadikurniawan.sucipto/projects/3D-CNN-VswinFormer/run.py
REPO_DIR="$(dirname "$RUN_SCRIPT")"

OUTPUT_DIR="$BASE_OUTPUT_DIR/job_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

exec > >(tee "$OUTPUT_DIR/slurm.log")
exec 2>&1

module purge
module load GCC/12.3.0

source /home/VIB.LOCAL/lunkyadikurniawan.sucipto/miniconda3/etc/profile.d/conda.sh
conda activate mri_3D_classification_env

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

COMMAND=(
    python "$RUN_SCRIPT"
    --output_dir "$OUTPUT_DIR"
    --epochs "$EPOCHS"
    --dataset "$DATASET"
)

if [ -n "$CHECKPOINT_PATH" ]; then
    COMMAND+=(--checkpoint_path "$CHECKPOINT_PATH")
fi

cat > "$OUTPUT_DIR/run_setup.txt" <<EOF
date: $(date)
hostname: $(hostname)

slurm_job_id: ${SLURM_JOB_ID:-}
slurm_job_name: ${SLURM_JOB_NAME:-}
slurm_partition: ${SLURM_JOB_PARTITION:-}
slurm_cpus_per_task: ${SLURM_CPUS_PER_TASK:-}
slurm_mem_per_node: ${SLURM_MEM_PER_NODE:-}
cuda_visible_devices: ${CUDA_VISIBLE_DEVICES:-}

base_output_dir: $BASE_OUTPUT_DIR
output_dir: $OUTPUT_DIR
epochs: $EPOCHS
run_script: $RUN_SCRIPT
checkpoint_path: ${CHECKPOINT_PATH:-None}
dataset: $DATASET
command: "${COMMAND[@]}"

git_branch: $(git -C "$REPO_DIR" branch --show-current)
git_commit: $(git -C "$REPO_DIR" rev-parse HEAD)
EOF

"${COMMAND[@]}"