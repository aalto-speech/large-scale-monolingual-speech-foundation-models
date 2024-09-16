#!/bin/bash
#SBATCH --job-name=prepare_fairseq_manifest_valid
#SBATCH --output=/prepare_fairseq_manifest_valid_radio_station_1.o
#SBATCH --error=/prepare_fairseq_manifest_valid_radio_station_1.e
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=0-12:00:00
#SBATCH --account=project_462000187

module load LUMI/23.09
module load FFmpeg/6.0-cpeGNU-23.09
start_time=$(date +%s.%N)
channel=radio_station_1
valid_size_hours=2
path_to_manifests=/fairseq_manifests_tar
mkdir -p ${path_to_manifests}/${channel}
source /projappl/project_462000187/python_envs/vad_env/bin/activate
python /scripts/data_preprocessing/prepare_fairseq_manifest_valid_tsv.py ${path_to_manifests} $channel ${valid_size_hours}
end_time=$(date +%s.%N)
elapsed_time_seconds=$(echo "$end_time - $start_time" | bc)
elapsed_time_minutes=$(echo "scale=2; $elapsed_time_seconds / 60" | bc)
echo "Elapsed time: $elapsed_time_minutes minutes"
