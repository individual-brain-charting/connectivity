#!/bin/bash
#
#SBATCH --job-name=permute
#SBATCH -c50
#SBATCH --partition=parietal,normal
#SBATCH --error=logs/jobid_%A.err
#SBATCH --output=logs/jobid_%A.out

# Run the Python script with the extracted arguments
srun python /data/parietal/store3/work/haggarwa/connectivity/scripts/supplementary/permutation_testing.py 0.2 across Runs "Ledoit-Wolf correlation"