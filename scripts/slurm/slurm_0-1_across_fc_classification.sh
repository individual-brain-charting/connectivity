#!/bin/bash
#SBATCH --job-name=0.1_across
#SBATCH -c20
#SBATCH --output=logs/across_0.1_jobid_%A_%a.out
#SBATCH --error=logs/across_0.1_jobid_%A_%a.err
#SBATCH --partition=normal,parietal

srun python /data/parietal/store3/work/haggarwa/connectivity/scripts/supplementary/classify_fc_var_lowpass.py 0.1 across