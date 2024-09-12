# Large-scale monolingual speech foundation models
Scripts for training large-scale monolingual speech foundation models with 158K hours of Finnish speech

## Pre-trained and fine-tuned (4,600 hours) models

Model
|---
[wav2vec 2.0 Base Pre-trained](TODO) 
[wav2vec 2.0 Base Fine-tuned](TODO) 
[wav2vec 2.0 Large Pre-trained](TODO) 
[wav2vec 2.0 Large Fine-tuned](TODO) 
[wav2vec 2.0 X-Large Pre-trained](TODO)
[wav2vec 2.0 X-Large Fine-tuned](TODO)

More details on the models are available in the [paper](TODO).
The models are also available at [Huggingface Hub](https://huggingface.co/collections/GetmanY1/wav2vec2-fi-150k-66c9d75d18579088974ea37f)

## Data pre-processing 

## Pre-training the models

The scripts shared in this repository are adapted to the AMD hardware of the [LUMI supercomputer](https://www.lumi-supercomputer.eu/). To train a wav2vec 2.0 Base model, run

```
sbatch /scripts/pretraining/pretrain_wav2vec2_base.sh
```

Note: you can simulate 512 GPUs by using k GPUs and adding command line parameters (before `--config-dir`)
`distributed_training.distributed_world_size=k` `+optimization.update_freq='[x]'` where x = 512/k

## Fine-tuning the models with CTC

To fine-tune a wav2vec 2.0 Base model using Fairseq, run

```
sbatch scripts/finetuning/full-scale-asr/finetune_wav2vec2_base.sh
```

* When pre-training on the LUMI supercomputer using Fairseq, it is crucial to set `export MIOPEN_FIND_MODE=2`. MIOpen is AMD’s deep-learning primitives library for GPUs (counterpart of NVIDIA's cuDNN). Setting the Find Mode to `2`, or `FAST' is crucial for optimal pre-training speed, otherwise pre-training is 10-20x times slower. More details on MIOpen Find modes are available [here](https://rocm.docs.amd.com/projects/MIOpen/en/docs-5.6.0/find_and_immediate.html) 

* You can simulate 128 GPUs by using k GPUs and adding command line parameters (before `--config-dir`)
`distributed_training.distributed_world_size=k` `+optimization.update_freq='[x]'` where x = 128/k

* For more LUMI-specific details on training with AMD GPUs, see [here](https://lumi-supercomputer.github.io/LUMI-training-materials/4day-20231003/extra_4_10_Best_Practices_GPU_Optimization/), [here](https://lumi-supercomputer.github.io/LUMI-EasyBuild-docs/p/PyTorch/), and [here](https://462000265.lumidata.eu/ai-20240529/files/LUMI-ai-20240529-10-Extreme_scale_AI.pdf).

## Fine-tuning the models with CTC using 🤗Transformers

To fine-tune a wav2vec 2.0 Base model using Huggingface Transformers, run

```
sbatch scripts/finetuning/low-resource-asr/finetune_wav2vec2_base.sh
```

## Computing Layer Utilization Rate (LUR)

![LUR](figures/ig_analysis.svg)

To calculate the neuron attributions using Integrated Gradients (IG), run `scripts/interpretation/ig_single_layer.sh` for each layer. After that, run the notebook `scripts/interpretation/compute_LUR.ipynb` to visualize the Layer Utilization Rates (LURs).

More details on the LUR are available in the [paper](TODO).
