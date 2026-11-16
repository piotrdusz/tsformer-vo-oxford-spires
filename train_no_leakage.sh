#!/bin/bash
#
#SBATCH --job-name=TSformer
#SBATCH --output=log_no_leakage.out
#SBATCH --time=24:00:00
#SBATCH --account=ai-air
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#

source .venv/bin/activate
python train.py --config configs/test_church.json