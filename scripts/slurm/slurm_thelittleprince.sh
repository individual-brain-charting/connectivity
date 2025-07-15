#!/bin/bash
#
#SBATCH --job-name=thelittleprince_classify_fc
#SBATCH -c20
#SBATCH --partition=parietal,normal
#SBATCH --error error_%A_%a.out

srun python /data/parietal/store3/work/haggarwa/connectivity/scripts/classify_fc_thelittleprince.py