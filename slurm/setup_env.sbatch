#!/usr/bin/env bash
#SBATCH --job-name=vla-setup
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-setup-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
bash setup_env.sh
