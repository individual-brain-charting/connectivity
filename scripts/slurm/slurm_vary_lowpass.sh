#!/bin/bash
#SBATCH --job-name=vary_lowpass
#SBATCH --output=log_slurm/jobid_%A_%a.out 
#SBATCH --error=log_slurm/jobid_%A_%a.err
#SBATCH --partition=normal,parietal
#SBATCH --ntasks-per-node=20
#SBATCH --time=48:00:00
#SBATCH --array=1-5

low_pass_values=(0.1 0.3 0.4 0.5)
low_pass=${low_pass_values[$SLURM_ARRAY_TASK_ID - 1]}

conda activate connpy

srun python /data/parietal/store3/work/haggarwa/connectivity/scripts/estimate_fc.py $low_pass