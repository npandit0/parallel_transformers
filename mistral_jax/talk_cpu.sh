#!/usr/bin/bash
#SBATCH --job-name=talk_cpu
#SBATCH --error=talk_cpu_%j_%a.err
#SBATCH --out=talk_cpu_%j_%a.out
#SBATCH --time=23:59:59
#SBATCH --mail-type=ALL
#SBATCH --partition=bigmem
#SBATCH --mem=256G  # minimum for bigmem

export JAX_PLATFORM_NAME=cpu
$SCRATCH/parallel_transformers/venvs/mistral_jax/bin/python the_model_speaks.py $@