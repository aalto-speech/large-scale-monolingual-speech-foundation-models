#!/bin/bash
#SBATCH --job-name=deduplicate
#SBATCH --output=/remove_empty_and_deduplicate.out
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=7 
#SBATCH --time=00-03:00:00
#SBATCH --account=project_462000971

module load LUMI/24.03 partition/C

source /scratch/project_462000857/python_envs/fairseq_env_2025/bin/activate

start_time=$(date +%s.%N)

python /data_preprocessing/remove_empty_and_deduplicate.py

end_time=$(date +%s.%N)
elapsed_time_seconds=$(echo "$end_time - $start_time" | bc)
elapsed_time_minutes=$(echo "scale=2; $elapsed_time_seconds / 60" | bc)
echo "Elapsed time: $elapsed_time_minutes minutes"
