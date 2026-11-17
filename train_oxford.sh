#!/bin/bash
#
#SBATCH --job-name=TSformer-Oxford
#SBATCH --output=log_oxford.out
#SBATCH --time=24:00:00
#SBATCH --account=ai-air
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#

set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

source .venv/bin/activate
python train.py --config configs/oxford_experiment.json