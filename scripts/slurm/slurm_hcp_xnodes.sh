#!/bin/bash
#
#SBATCH --job-name=hcp_classify_fc
#SBATCH -c63
#SBATCH --partition=parietal,normal
#SBATCH --error=logs/jobid_%A_%a.err
#SBATCH --output=logs/jobid_%A_%a.out
#SBATCH --array=1-63

# Get the specific line from tmp_combinations.txt based on the array task ID
COMBINATIONS_FILE="/data/parietal/store3/work/haggarwa/connectivity/data/hcp_combinations.txt"
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $COMBINATIONS_FILE)

# Parse the line to extract task1, task2, and connectivity_measure
read -r TASK1 TASK2 CONNECTIVITY_MEASURE <<< "$LINE"

# Run the Python script with the extracted arguments
srun python /data/parietal/store3/work/haggarwa/connectivity/scripts/supplementary/classify_subjects_hcp_slurm_compatible.py "$TASK1" "$TASK2" "$CONNECTIVITY_MEASURE"