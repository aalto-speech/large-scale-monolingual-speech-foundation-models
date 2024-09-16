#!/bin/bash
#SBATCH --job-name=vad_2024   # Job name
#SBATCH --output=/segment_radio_station_1_with_vad_and_tar_2024.o
#SBATCH --error=/segment_radio_station_1_with_vad_and_tar_2024.e
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=20G
#SBATCH --gpus-per-node=1
#SBATCH --time=1-23:59:00
#SBATCH --account=project_462000187

module purge
module load CrayEnv
module load PrgEnv-cray/8.3.3
module load craype-accel-amd-gfx90a
module load cray-python
module use /pfs/lustrep2/projappl/project_462000125/samantao-public/mymodules
module load aws-ofi-rccl/rocm-5.2.3.lua
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_DEBUG=INFO
export RDZV_PORT=29400
export NCCL_NET_GDR_LEVEL=3
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-cache-${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
export CXI_FORK_SAFE=1
export CXI_FORK_SAFE_HP=1
export FI_CXI_DISABLE_CQ_HUGETLB=1

if [ $SLURM_LOCALID -eq 0 ] ; then
    rm -rf $MIOPEN_USER_DB_PATH
    mkdir -p $MIOPEN_USER_DB_PATH
fi
export MIOPEN_FIND_MODE=2

start_time=$(date +%s.%N)
cd /radio_station_1_flac_segmented
mkdir -p /radio_station_1_tar
source /my_python_env/bin/activate

year=2024

for path in $(find /radio_station_1_flac/$year -maxdepth 1 -mindepth 1 -type d -name "1*")
do
month=$(basename $path)
python /scripts/data_preprocessing/run_vad_one_month.py $year $month
tar -rf /radio_station_1_tar/radio_station_1_$year.tar $year/$month
echo "Radio station 1 $year $month"
num_files=$(find "/radio_station_1_flac_segmented/$year/$month" -type f -name "*.flac" | wc -l)
echo "Number of files found: $num_files"
rm -r /radio_station_1_flac_segmented/$year/$month
done

end_time=$(date +%s.%N)
elapsed_time_seconds=$(echo "$end_time - $start_time" | bc)
elapsed_time_minutes=$(echo "scale=2; $elapsed_time_seconds / 60" | bc)
echo "Elapsed time: $elapsed_time_minutes minutes"
