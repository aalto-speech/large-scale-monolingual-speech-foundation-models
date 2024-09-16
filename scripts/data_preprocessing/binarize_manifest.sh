#!/bin/bash
#SBATCH --job-name=binarize
#SBATCH --output=/binarize_manifest/binarize_manifest_radio_channel_1.o
#SBATCH --error=/binarize_manifest/binarize_manifest_radio_channel_1.e
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=0-02:30:00
#SBATCH --account=project_462000187

module purge
module load CrayEnv
module load PrgEnv-cray/8.3.3
module load cray-python

source /my_python_env/bin/activate

channel=radio_channel_1
cd /binarize_manifest
dest_dir=/binarize_manifest/manifest_bin_$channel
train_split=train_$channel
valid_split=valid_$channel
fairseq_root=/my_python_env/fairseq
bash ${fairseq_root}/examples/wav2vec/scripts/binarize_manifest.sh ${dest_dir} ${train_split} ${valid_split} ${fairseq_root}
