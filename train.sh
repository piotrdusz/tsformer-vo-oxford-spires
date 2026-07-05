#!/bin/bash
#
#SBATCH --job-name=TSformer
#SBATCH --output=log.out
#SBATCH --time=04:00:00
#SBATCH --account=ai-air
#SBATCH --partition=short
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#

source .venv/bin/activate
python train.py
