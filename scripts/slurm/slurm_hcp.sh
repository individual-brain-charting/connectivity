#!/bin/bash
#
#SBATCH --job-name=hcp_classify_fc
#SBATCH -c50
#SBATCH --partition=parietal,normal
#SBATCH --error=jobid_%A.err
#SBATCH --output=jobid_%A.out

srun python /data/parietal/store3/work/haggarwa/connectivity/scripts/supplementary/classify_subjects_hcp.py