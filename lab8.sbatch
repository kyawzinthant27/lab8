#!/bin/bash
#SBATCH --job-name=lab8
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --partition=gpu3080
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$HOME/deep_learning/env_dl/bin/activate"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Started: $(date)"
echo "Python: $(which python)"
python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable in this GPU allocation'"
srun python -u lab8.py
echo "Finished: $(date)"
