#!/bin/bash
#SBATCH --partition=standard-g
#SBATCH --nodes=8
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --cpus-per-task=7 
#SBATCH --gpus-per-node=8
#SBATCH --time=00-12:00:00
#SBATCH --account=project_462000971

c=fe
MYMASKS="0x${c}000000000000,0x${c}00000000000000,0x${c}0000,0x${c}000000,0x${c},0x${c}00,0x${c}00000000,0x${c}0000000000"
module purge
module load CrayEnv
module load PrgEnv-cray #/8.3.3
module load craype-accel-amd-gfx90a
module load cray-python/3.10.10
module load gcc-native

module load LUMI/24.03 partition/G
module load rocm/6.0.3
module load aws-ofi-rccl/17d41cb-cpeGNU-24.03
module list

source /scratch/project_462000971/python_envs/fairseq_env_08_2025/bin/activate

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

input_tsv=${path_to_input_manifests}/${tv_radio}/${station}_train_and_valid.tsv
output_tsv=${path_to_output_manifests}/${tv_radio}/${station}_train_and_valid.tsv

srun --nodes=$SLURM_JOB_NUM_NODES --ntasks-per-node=8 --gpus-per-node=8 --cpus-per-task=7 $(which python) ${path_to_scripts}/run_wav2vec2_asr.py \
    --input_tsv=$input_tsv \
    --output_tsv=$output_tsv \
    --input_tsv_header=$path_to_tars \
    --tv_radio=$tv_radio \
    --station=$station \
    --shared_storage=/scratch/project_462000971/home/getmanya/shared_temp/${tv_radio}/$station

end_time=$(date +%s.%N)
elapsed_time_seconds=$(echo "$end_time - $start_time" | bc)
elapsed_time_minutes=$(echo "scale=2; $elapsed_time_seconds / 60" | bc)
echo "Elapsed time: $elapsed_time_minutes minutes"
