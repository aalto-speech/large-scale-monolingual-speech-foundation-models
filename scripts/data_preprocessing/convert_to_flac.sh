#!/bin/bash
#SBATCH --job-name=convert_2024
#SBATCH --output=/convert_to_flac_radio_station_1_2024.o
#SBATCH --error=/convert_to_flac_radio_station_1_2024.e
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=0-15:00:00
#SBATCH --account=project_462000187

module load LUMI/23.09
module load FFmpeg/6.0-cpeGNU-23.09

start_time=$(date +%s.%N)

year=2024
folder=raw_tv_and_radio_data/radio_channel_1/$year

for filepath in $(find "$folder" -type f -name "*.ts")
do
newfilepath=${filepath/radio_channel_1/radio_channel_1_flac}
mkdir -p $(dirname "$newfilepath")
ffmpeg -i $filepath -c:a flac -sample_fmt s16 -ar 16000 -ac 1 -vn -dn -sn -ignore_unknown \
 ${newfilepath%.ts}.flac
done

end_time=$(date +%s.%N)

num_files=$(find "$folder" -type f -name "*.ts" | wc -l)
echo "Number of files found: $num_files"
elapsed_time_seconds=$(echo "$end_time - $start_time" | bc)
elapsed_time_minutes=$(echo "scale=2; $elapsed_time_seconds / 60" | bc)
echo "Elapsed time: $elapsed_time_minutes minutes"
