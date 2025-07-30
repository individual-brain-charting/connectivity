#!/bin/bash
#SBATCH --job-name=0.3_lowpass
#SBATCH --output=logs/lowpass_0.3_jobid_%A_%a.out 
#SBATCH --error=logs/lowpass_0.3_jobid_%A_%a.err
#SBATCH --partition=normal,parietal

srun python /data/parietal/store3/work/haggarwa/connectivity/scripts/estimate_fc.py 0.3