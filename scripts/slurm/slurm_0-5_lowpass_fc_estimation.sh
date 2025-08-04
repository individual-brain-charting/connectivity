#!/bin/bash
#SBATCH --job-name=0.5_lowpass
#SBATCH -c20
#SBATCH --output=logs/lowpass_0.5_jobid_%A_%a.out
#SBATCH --error=logs/lowpass_0.5_jobid_%A_%a.err
#SBATCH --partition=normal,parietal

srun python /data/parietal/store3/work/haggarwa/connectivity/scripts/supplementary/estimate_fc_var_lowpass.py 0.5
