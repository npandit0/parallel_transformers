#!/usr/bin/bash
#SBATCH --job-name=dpm_gpu
#SBATCH --error=dpm_gpu_%j_%a.err
#SBATCH --out=dpm_gpu_%j_%a.out
#SBATCH --time=01:59:59
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mail-type=ALL

$SCRATCH/parallel_transformers/venvs/mistral_jax/bin/python deer_prototype_mistral.py $@