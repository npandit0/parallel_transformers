#!/usr/bin/bash
#SBATCH --job-name=dpm_cpu
#SBATCH --error=dpm_cpu_%j_%a.err
#SBATCH --out=dpm_cpu_%j_%a.out
#SBATCH --time=00:59:59
#SBATCH --mail-type=ALL

$SCRATCH/parallel_transformers/venvs/mistral_jax/bin/python deer_prototype_mistral.py $@ --xavier