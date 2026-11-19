#!/bin/bash
#SBATCH --job-name=finetune_wav2vec2_base
#SBATCH --output=/finetune_wav2vec2_base.o
#SBATCH --error=/finetune_wav2vec2_base.e
#SBATCH --partition=standard-g
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=0-03:00:00
#SBATCH --account=project_462000971

c=fe
MYMASKS="0x${c}000000000000,0x${c}00000000000000,0x${c}0000,0x${c}000000,0x${c},0x${c}00,0x${c}00000000,0x${c}0000000000"
module purge
module load CrayEnv
module load PrgEnv-cray
module load craype-accel-amd-gfx90a
module load cray-python/3.10.10
module load gcc-native

module load LUMI/24.03 partition/G
module load rocm/6.0.3
module load aws-ofi-rccl/17d41cb-cpeGNU-24.03
module list

if [ ! -d /my_python_env ] ; then
    python -m venv --system-site-packages my_python_env
    source my_python_env/bin/activate
    cd my_python_env

    pip install git+https://github.com/Getmany1/omegaconf@2.0_branch
    pip install hydra-core==1.0.7
    
    pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/rocm6.0
    
    pip install tqdm timm librosa pyarrow cffi ninja cython scikit-learn ipython packaging wheel regex sacrebleu bitarray tensorboardx python-hostlist wandb editdistance jiwer datasets transformers accelerate fairseq fairscale

    git clone https://github.com/Getmany1/fairseq
    cd fairseq
    pip install --no-deps --no-build-isolation --editable ./ --verbose    # if this one fails, try     pip install --editable ./ --verbose
    python setup.py build develop
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
    dataset.num_workers=7 \
    dataset.max_tokens=2400000 \
    distributed_training.distributed_world_size=128 \
    distributed_training.nprocs_per_node=8 \
    distributed_training.distributed_port=$RDZV_PORT \
    +model.w2v_path=/my_pretrained_wav2vec2.pt \
    --config-dir config/finetuning \
    --config-name  wav2vec2_base_lp_finparl.yaml
