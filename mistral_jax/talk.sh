#!/usr/bin/bash
#SBATCH --job-name=talk
#SBATCH --error=talk_%j_%a.err
#SBATCH --out=talk_%j_%a.out
#SBATCH --time=23:59:59
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mail-type=ALL

$SCRATCH/parallel_transformers/venvs/mistral_jax/bin/python the_model_speaks.py $@ --load_weights --parallel