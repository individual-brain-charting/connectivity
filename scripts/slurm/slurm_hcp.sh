#!/bin/bash
#
#SBATCH --job-name=hcp_classify_fc
#SBATCH -c40
#SBATCH --partition=parietal,normal
#SBATCH --error jobid_%A.err
#SBATCH --output=logs/jobid_%A.out

srun python /data/parietal/store3/work/haggarwa/connectivity/scripts/supplementary/classify_fc_hcp.py