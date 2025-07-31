#!/bin/bash
#SBATCH --job-name=0.1_within
#SBATCH --output=logs/within_0.1_jobid_%A_%a.out
#SBATCH --error=logs/within_0.1_jobid_%A_%a.err
#SBATCH --partition=normal,parietal

srun python /data/parietal/store3/work/haggarwa/connectivity/scripts/supplementary/classify_fc_var_lowpass.py 0.1 within