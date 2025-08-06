#!/bin/bash
#SBATCH --job-name=nyquist_across
#SBATCH -c20
#SBATCH --output=logs/across_nyquist_jobid_%A_%a.out
#SBATCH --error=logs/across_nyquist_jobid_%A_%a.err
#SBATCH --partition=parietal

srun python /data/parietal/store3/work/haggarwa/connectivity/scripts/supplementary/classify_fc_var_lowpass.py nyquist across