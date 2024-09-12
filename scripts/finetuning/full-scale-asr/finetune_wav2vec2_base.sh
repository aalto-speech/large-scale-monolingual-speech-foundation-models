#!/bin/bash
#SBATCH --job-name=finetune_wav2vec2_base
#SBATCH --output=/finetune_wav2vec2_base.o
#SBATCH --error=/finetune_wav2vec2_base.e
#SBATCH --partition=dev-g
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=0-03:00:00
#SBATCH --account=project_462000187

c=fe
MYMASKS="0x${c}000000000000,0x${c}00000000000000,0x${c}0000,0x${c}000000,0x${c},0x${c}00,0x${c}00000000,0x${c}0000000000"

module purge
module load CrayEnv
module load PrgEnv-cray/8.3.3
module load craype-accel-amd-gfx90a
module load cray-python

module use /pfs/lustrep2/projappl/project_462000125/samantao-public/mymodules
module load aws-ofi-rccl/rocm-5.2.3.lua
module list

if [ ! -d /my_python_env ] ; then
    python -m venv --system-site-packages my_python_env
    source my_python_env/bin/activate
    cd my_python_env

    pip install git+https://github.com/Getmany1/omegaconf@2.0_branch
    
    pip install torch==1.13.1+rocm5.2 torchvision==0.14.1+rocm5.2 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/rocm5.2
    pip install librosa pyarrow datasets transformers accelerate timm fairseq fairscale wandb python-hostlist tensorboardx
    cp /scripts/misc/corrected_fairseq_utils.py lib/python3.10/site-packages/fairseq/distributed/utils.py
    
    git clone https://github.com/ROCmSoftwarePlatform/apex
    cd apex
    module load aws-ofi-rccl/rocm-5.5.0.lua
    python setup.py install --cpp_ext --cuda_ext
    module load aws-ofi-rccl/rocm-5.2.3.lua
else
    source /my_python_env/bin/activate
fi

export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_DEBUG=INFO
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

srun -N 16 --gpus 128 --ntasks-per-node=8 --cpus-per-task=7 --cpu-bind=mask_cpu:$MYMASKS \
fairseq-hydra-train \
    +model.w2v_path=/my_pretrained_wav2vec2.pt \
    --config-dir config/finetuning \
    --config-name  wav2vec2_base_lp_finparl.yaml
