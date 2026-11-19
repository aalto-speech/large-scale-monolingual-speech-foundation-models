#!/bin/bash
#SBATCH --partition=small
#SBATCH --output=/asr_prob_-01_train.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 
#SBATCH --ntasks=1
#SBATCH --mem=150G
#SBATCH --cpus-per-task=8
#SBATCH --time=0-03:00:00
#SBATCH --account=project_462000857

module load LUMI/24.03 partition/C
start_time=$(date +%s.%N)

path_to_tsvs=/tsv_vad_only_with_asr_deduplicated_nonempty

source /scratch/project_462000857/python_envs/fairseq_env_2025/bin/activate
python /data_preprocessing/prepare_fairseq_manifest_filter_by_condition.py \
  --input $path_to_tsvs/train_and_valid/train_FULL.tsv \
  --output "$path_to_tsvs/filtered/train_asrprob-01_FULL.tsv" \
  --condition "non_blank_log_prob_mean >= -0.1"

ln -s "$path_to_tsvs/filtered/train_asrprob-005_FULL.tsv" /binarize_manifest_filtered/
end_time=$(date +%s.%N)
elapsed_time_seconds=$(echo "$end_time - $start_time" | bc)
elapsed_time_minutes=$(echo "scale=2; $elapsed_time_seconds / 60" | bc)
echo "Elapsed time: $elapsed_time_minutes minutes"
