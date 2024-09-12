#!/bin/bash -l
#SBATCH --job-name=ig_layer_1_wav2vec2_base
#SBATCH --output=/ig_layer_1_wav2vec2_base.o
#SBATCH --error=/ig_layer_1_wav2vec2_base.e
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00
#SBATCH --account=project_462000187

module purge
module load CrayEnv
module load PrgEnv-cray/8.3.3
module load craype-accel-amd-gfx90a
module load cray-python

module use /pfs/lustrep2/projappl/project_462000125/samantao-public/mymodules
module load aws-ofi-rccl/rocm-5.2.3.lua

source /my_python_env/bin/activate

layernum=1
python /scripts/interpretation/ig_single_layer.py -l $layernum
