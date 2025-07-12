#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=1
#SBATCH --mem=100M
#SBATCH --time=00:05:00
#SBATCH --job-name=parallel_run

srun bash -c 'python run_from_terminal.py --param=$SLURM_PROCID'