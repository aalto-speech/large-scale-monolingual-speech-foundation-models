#!/bin/bash
c=fe
MYMASKS="0x${c}000000000000,0x${c}00000000000000,0x${c}0000,0x${c}000000,0x${c},0x${c}00,0x${c}00000000,0x${c}0000000000"

source /my_python_env/bin/activate

export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_DEBUG=INFO
export RDZV_PORT=29400
export NCCL_NET_GDR_LEVEL=3
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-cache-${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
export CXI_FORK_SAFE=1
export CXI_FORK_SAFE_HP=1
export FI_CXI_DISABLE_CQ_HUGETLB=1

export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_GPU_IPC_ENABLED=1

if [ $SLURM_LOCALID -eq 0 ] ; then
    rm -rf $MIOPEN_USER_DB_PATH
    mkdir -p $MIOPEN_USER_DB_PATH
fi

export MIOPEN_FIND_MODE=2
rocm-smi
sleep 2

fairseq-hydra-train \
dataset.num_workers=0 \
dataset.max_tokens=1800000 \
distributed_training.distributed_world_size=512 \
distributed_training.nprocs_per_node=8 \
distributed_training.distributed_port=$RDZV_PORT \
--config-dir /config/pretraining \
--config-name wav2vec2_large_kavi.yaml
