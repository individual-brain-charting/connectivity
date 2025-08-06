#!/bin/bash
#SBATCH --job-name=nyquist_within
#SBATCH -c20
#SBATCH --output=logs/within_nyquist_jobid_%A_%a.out
#SBATCH --error=logs/within_nyquist_jobid_%A_%a.err
#SBATCH --partition=normal,parietal

srun python /data/parietal/store3/work/haggarwa/connectivity/scripts/supplementary/classify_fc_var_lowpass.py nyquist within