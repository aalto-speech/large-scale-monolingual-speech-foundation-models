#!/bin/bash -l
#SBATCH --job-name=finetune_wav2vec2_large
#SBATCH --output=/finetune_wav2vec2_large.o
#SBATCH --error=/finetune_wav2vec2_large.e
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=8
#SBATCH --time=0-06:00:00
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

torchrun \
	--nproc_per_node 8 /scripts/finetuning/low-resource-asr/run_speech_recognition_ctc_multigpu.py \
	--dataset_name="cv_16_1" \ # --dataset_name="fleurs_fi" \
  --cache_dir="/cache" \
	--model_name_or_path="/pretrained_wav2vec2_hf" \
  --tokenizer_name_or_path="/pretrained_wav2vec2_hf" \
	--output_dir="/outputs_hf" \
	--report_to="wandb" \
	--num_train_epochs="80" \
  --preprocessing_num_workers="$(nproc)" \
	--per_device_train_batch_size="24" \
	--learning_rate="5e-5" \
	--warmup_ratio="0.25" \
	--evaluation_strategy="epoch" \
  --logging_strategy="epoch" \
  --save_strategy="epoch" \
	--text_column_name="text" \
	--save_total_limit="6" \
  --eval_metrics wer cer \
	--load_best_model_at_end=True \
  --metric_for_best_model="wer" \
   --greater_is_better=False \
	--freeze_feature_encoder \
	--gradient_checkpointing \
	--chars_to_ignore , ? . ! - \; \: \" % ‘ � \
	--fp16 \
	--group_by_length \
  --ddp_timeout="18000" \
	--do_train --do_eval
