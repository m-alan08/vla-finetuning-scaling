#!/usr/bin/env bash
#SBATCH --job-name=vla-probe
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --exclude=nvidia-gh200-[1-4],amd-mi210-[1-4],amd-mi300a-[1-2],nvidia-p40-[1-8],dgx1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-probe-%j.out

set -uo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
if [ ! -f probe/probe.py ]; then
  echo "Submit this job from the repository root (probe/probe.py not found in $PWD)." >&2
  exit 2
fi

source slurm/env.sh

if [ ! -x "$VLA_VENV/bin/python" ]; then
  echo "No Python environment at $VLA_VENV. Run 'bash setup_env.sh' first." >&2
  exit 2
fi

echo "job $SLURM_JOB_ID on $(hostname), partition ${SLURM_JOB_PARTITION:-?}, started $(date)"
nvidia-smi || echo "nvidia-smi not available"

python probe/probe.py --require-gpu "$@"
status=$?
echo "probe exit code $status, finished $(date)"
exit $status
