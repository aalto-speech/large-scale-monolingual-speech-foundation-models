# Finnish speech SSL at scale: 471K hours from 2.1M hours of broadcast audio (2026)

Scripts for training large-scale self-supervised speech foundation models on 471K hours of Finnish speech. The data was selected from 2.1M hours of raw broadcast audio (KAVI 2.0) through deduplication and ASR confidence filtering, as described in [From 2.1M Hours of Raw Broadcast Audio to Finnish Speech SSL: Confidence Filtering and Architectural Analysis at Scale (Getman et al., 2026)](https://arxiv.org/abs/2610.13102).

## Pretrained and fine-tuned models
 
We release 14 pretrained models (Base, Large and X-Large, each with Transformer, Conformer and E-Branchformer encoders of matched size) and their versions fine-tuned for Finnish ASR on 4.6K hours of transcribed speech.
 
- **Hugging Face Hub:** [wav2vec2-fi-471k](https://huggingface.co/collections/GetmanY1/wav2vec2-fi-471k). Each repository contains the 🤗 Transformers model and the original fairseq checkpoint. Conformer and E-Branchformer models use custom modeling code, so load them with `trust_remote_code=True`.
- **Zenodo:** [fairseq checkpoints](https://zenodo.org/records/24504711) of all 28 models.
- **Filtering ablation:** [wav2vec2-base-fi-filtering-ablation](https://huggingface.co/collections/GetmanY1/wav2vec2-base-fi-filtering-ablation), the Base models from Figure 1 of the paper, pretrained on data selected with different filtering criteria.
Test WER (%) after fine-tuning (Table III of the paper). LP = Lahjoita puhetta, FP = Finnish Parliament, CV = Common Voice 16.1, KAVI = real-world broadcast speech. Best results per scale in bold.
 
| Model | Params | LP | FP'16 | FP'20 | CV | FLEURS | KAVI |
|---|---|---|---|---|---|---|---|
| ***Base*** | | | | | | | |
| Transformer, 12L, 158K h baseline ([pretrained](https://huggingface.co/GetmanY1/wav2vec2-base-fi-150k), [fine-tuned](https://huggingface.co/GetmanY1/wav2vec2-base-fi-150k-finetuned)) | 95.0M | 22.27 | 10.92 | 8.42 | 10.61 | 13.27 | 27.45 |
| Transformer, 12L ([pretrained](https://huggingface.co/GetmanY1/wav2vec2-base-12l-fi-471k), [fine-tuned](https://huggingface.co/GetmanY1/wav2vec2-base-12l-fi-471k-finetuned)) | 95.0M | 22.28 | 10.72 | 8.70 | 9.74 | 13.02 | 24.85 |
| Transformer, 18L ([pretrained](https://huggingface.co/GetmanY1/wav2vec2-base-18l-fi-471k), [fine-tuned](https://huggingface.co/GetmanY1/wav2vec2-base-18l-fi-471k-finetuned)) | 95.1M | 21.33 | 10.64 | 8.66 | 8.99 | 12.71 | 23.91 |
| Conformer, 12L ([pretrained](https://huggingface.co/GetmanY1/wav2vec2-base-conformer-12l-fi-471k), [fine-tuned](https://huggingface.co/GetmanY1/wav2vec2-base-conformer-12l-fi-471k-finetuned)) | 96.7M | 18.89 | 10.08 | 8.07 | 6.46 | 11.20 | 23.71 |
| Conformer, 17L ([pretrained](https://huggingface.co/GetmanY1/wav2vec2-base-conformer-17l-fi-471k), [fine-tuned](https://huggingface.co/GetmanY1/wav2vec2-base-conformer-17l-fi-471k-finetuned)) | 96.5M | **18.50** | 9.83 | 7.94 | **5.69** | **10.56** | 22.71 |
| E-Branchformer, 12L ([pretrained](https://huggingface.co/GetmanY1/wav2vec2-base-ebranch-12l-fi-471k), [fine-tuned](https://huggingface.co/GetmanY1/wav2vec2-base-ebranch-12l-fi-471k-finetuned)) | 98.4M | 19.14 | 10.10 | **7.80** | 6.91 | 11.53 | 22.89 |
| E-Branchformer, 17L ([pretrained](https://huggingface.co/GetmanY1/wav2vec2-base-ebranch-17l-fi-471k), [fine-tuned](https://huggingface.co/GetmanY1/wav2vec2-base-ebranch-17l-fi-471k-finetuned)) | 96.6M | 18.80 | **9.54** | 7.85 | 6.19 | 11.09 | **22.44** |
| ***Large*** | | | | | | | |
| Transformer, 24L, 158K h baseline ([pretrained](https://huggingface.co/GetmanY1/wav2vec2-large-fi-150k), [fine-tuned](https://huggingface.co/GetmanY1/wav2vec2-large-fi-150k-finetuned)) | 317.4M | 16.86 | 8.29 | 6.94 | 5.86 | 10.69 | 23.60 |
| Transformer, 24L ([pretrained](https://huggingface.co/GetmanY1/wav2vec2-large-24l-fi-471k), [fine-tuned](https://huggingface.co/GetmanY1/wav2vec2-large-24l-fi-471k-finetuned)) | 317.4M | 16.78 | 8.41 | 6.77 | 6.20 | 10.42 | 21.02 |
| Conformer, 24L ([pretrained](https://huggingface.co/GetmanY1/wav2vec2-large-conformer-24l-fi-471k), [fine-tuned](https://huggingface.co/GetmanY1/wav2vec2-large-conformer-24l-fi-471k-finetuned)) | 317.2M | 16.46 | 8.58 | **6.71** | 4.52 | 9.69 | **19.86** |
| Conformer, 36L ([pretrained](https://huggingface.co/GetmanY1/wav2vec2-large-conformer-36l-fi-471k), [fine-tuned](https://huggingface.co/GetmanY1/wav2vec2-large-conformer-36l-fi-471k-finetuned)) | 317.1M | 16.36 | 8.16 | 6.81 | **4.28** | 9.43 | 21.02 |
| E-Branchformer, 24L ([pretrained](https://huggingface.co/GetmanY1/wav2vec2-large-ebranch-24l-fi-471k), [fine-tuned](https://huggingface.co/GetmanY1/wav2vec2-large-ebranch-24l-fi-471k-finetuned)) | 317.1M | 16.30 | 8.22 | 6.84 | 4.33 | 9.59 | 20.01 |
| E-Branchformer, 36L ([pretrained](https://huggingface.co/GetmanY1/wav2vec2-large-ebranch-36l-fi-471k), [fine-tuned](https://huggingface.co/GetmanY1/wav2vec2-large-ebranch-36l-fi-471k-finetuned)) | 317.4M | **16.11** | **8.15** | 6.88 | 5.24 | **9.30** | 20.09 |
| ***X-Large*** | | | | | | | |
| Transformer, 48L, 158K h baseline ([pretrained](https://huggingface.co/GetmanY1/wav2vec2-xlarge-fi-150k), [fine-tuned](https://huggingface.co/GetmanY1/wav2vec2-xlarge-fi-150k-finetuned)) | 965.5M | 16.37 | 7.81 | 6.43 | 5.42 | 9.96 | 22.42 |
| Transformer, 48L ([pretrained](https://huggingface.co/GetmanY1/wav2vec2-xlarge-48l-fi-471k), [fine-tuned](https://huggingface.co/GetmanY1/wav2vec2-xlarge-48l-fi-471k-finetuned)) | 965.5M | **16.27** | 7.83 | **6.26** | 3.85 | 9.37 | **19.54** |
| Conformer, 48L ([pretrained](https://huggingface.co/GetmanY1/wav2vec2-xlarge-conformer-48l-fi-471k), [fine-tuned](https://huggingface.co/GetmanY1/wav2vec2-xlarge-conformer-48l-fi-471k-finetuned)) | 964.7M | 17.13 | 7.78 | 6.45 | **3.68** | **9.08** | 20.75 |
| E-Branchformer, 48L ([pretrained](https://huggingface.co/GetmanY1/wav2vec2-xlarge-ebranch-48l-fi-471k), [fine-tuned](https://huggingface.co/GetmanY1/wav2vec2-xlarge-ebranch-48l-fi-471k-finetuned)) | 965.6M | 16.74 | **7.63** | 6.38 | 4.45 | 9.13 | 19.96 |

## Data preprocessing

Compared to [previous work](https://www.isca-archive.org/interspeech_2025/getman25_interspeech.html), the pipeline adds the following steps:

1. After voice activity detection (VAD), ASR transcripts and CTC log-probabilities are extracted for each segment.
2. Duplicate segments (e.g., rebroadcasts) are removed based on the ASR transcripts.
3. Segments are filtered by ASR confidence computed from the CTC log-probabilities.

Scripts for these steps are in `scripts/data_preprocessing/kavi_2_0`.

## Pretraining the models

wav2vec 2.0 models with alternative encoders in addition to the vanilla Transformer are trained, including the convolution-augmented Conformer and E-Branchformer. The latter is not natively supported in fairseq, so it is implemented in our fork at https://github.com/Getmany1/fairseq.

Pretraining configurations are in `config/pretraining/kavi_2_0`.

# Large-scale monolingual speech foundation models (2025)

Scripts for training self-supervised monolingual speech foundation models on 158K hours of Finnish speech, as described in [Is your model big enough? Training and interpreting large-scale monolingual speech foundation models (Getman et al., 2025)](https://www.isca-archive.org/interspeech_2025/getman25_interspeech.html).


## Pretrained and fine-tuned (4,600 hours) models

Model
|---
[wav2vec 2.0 Base pretrained](https://zenodo.org/doi/10.5281/zenodo.13754087) 
[wav2vec 2.0 Base Fine-tuned](https://zenodo.org/doi/10.5281/zenodo.13754085) 
[wav2vec 2.0 Large pretrained](https://zenodo.org/doi/10.5281/zenodo.13754083) 
[wav2vec 2.0 Large Fine-tuned](https://zenodo.org/doi/10.5281/zenodo.13754081) 
[wav2vec 2.0 X-Large pretrained](https://zenodo.org/doi/10.5281/zenodo.13754079)
[wav2vec 2.0 X-Large Fine-tuned](https://zenodo.org/doi/10.5281/zenodo.13754077)

More details on the models are available in the [paper](https://www.isca-archive.org/interspeech_2025/getman25_interspeech.html).
The models are also available at [Hugging Face Hub](https://huggingface.co/collections/GetmanY1/wav2vec2-fi-150k-66c9d75d18579088974ea37f)

## Training logs

Developing a foundation model from scratch requires not only vast amounts of unlabeled speech data but also substantial computational resources. Moreover, extensive hyperparameter search is often not feasible for large-scale models. Therefore, we are glad to share our pretraining logs on Weights & Biases (W&B) to provide more insights for other researchers developing their own speech foundation models.

* [wav2vec 2.0 Base](https://wandb.ai/getmanya/wav2vec2_base_KAVI?nw=nwusergetmanya)
* [wav2vec 2.0 Large](https://wandb.ai/getmanya/wav2vec2_large_KAVI?nw=nwusergetmanya)
* [wav2vec 2.0 X-Large](https://wandb.ai/getmanya/wav2vec2_1B_KAVI?nw=nwusergetmanya)

## Data preprocessing

The raw, unlabeled TV and radio data are organized into 1-hour files, each located in the directory `channel_name/year/month/day/channel_name_start_time-end_time.ts`:

```
.
└── raw_tv_and_radio_data/
    ├── radio_channel_1/
    │   ├── 2009/
    │   │   ├── 01/
    │   │   │   ├── 01/
    │   │   │   │   ├── radio_channel_1_0000-0100.ts
    │   │   │   │   ├── radio_channel_1_0100-0200.ts
    │   │   │   │   └── ...
    │   │   │   ├── 02/
    │   │   │   │   └── ...
    │   │   │   └── ...
    │   │   ├── 02/
    │   │   │   └── .../
    │   │   │       └── ...
    │   │   └── ...
    │   └── 2010/
    │       └── .../
    │           └── .../
    │               └── ...
    ├── tv_channel_2/
    │   └── ...
    └── ...
```

1. Convert the files to 16kHz mono flac audio by running `scripts/data_preprocessing/convert_to_flac.sh`. The script preserves the original folder structure.
2. Run voice activity detection (VAD) to split the data into shorter utterances and reduce the non-speech events, such as music, noise, and silence, and put them into _uncompressed_ (.tar) tarballs, with one archive per year per radio station or TV channel. The script `scripts/data_preprocessing/segment_with_vad_and_tar.sh` does it for one year of the data from `radio_station_1`. The script also stores a Python dictionary `out_file_to_nframes_dict` with the number of frames for each audio segment, which will be needed later to create the fairseq manifest of the data.
**Note:** fairseq does not support compressed archives
**Note:** Millions of small files affect the performance of any filesystem. As a result, quotas on Lustre filesystems are typically limited to several million files. To avoid running out of quota, put the short audio files into a .tar archive after VAD-based segmentation of a small part of the raw data (one day, month, or year), and remove them immediately afterward. You can also consider storing the preprocessed audio files in the `/tmp` folder, which usually does not consume the quota.
3. Prepare the fairseq manifest of the data. `scripts/data_preprocessing/prepare_fairseq_manifest.sh` creates a .tsv file with all `radio_station_1` audio samples stored in the corresponding .tar archives. To hold out a validation subset `valid_size_hours` hours, run `scripts/data_preprocessing/prepare_fairseq_manifest_valid_tsv.sh` afterward.
4. Binarize the fairseq manifest by running `scripts/data_preprocessing/binarize_manifest.sh`. This step is recommended for large datasets to avoid running out of RAM during pretraining. 

## Pretraining the models

The scripts shared in this repository are adapted to the AMD hardware of the [LUMI supercomputer](https://www.lumi-supercomputer.eu/). To train a wav2vec 2.0 Base model, run

```
sbatch /scripts/pretraining/pretrain_wav2vec2_base.sh
```

Note: you can simulate 512 GPUs by using k GPUs and adding command-line parameters (before `--config-dir`)
`distributed_training.distributed_world_size=k` `+optimization.update_freq='[x]'` where x = 512/k

## Fine-tuning the models with CTC

To fine-tune a wav2vec 2.0 Base model using fairseq, run

```
sbatch scripts/finetuning/full-scale-asr/finetune_wav2vec2_base.sh
```

* When pretraining on the LUMI supercomputer using fairseq, it is crucial to set `export MIOPEN_FIND_MODE=2`. MIOpen is AMD’s deep-learning primitives library for GPUs (counterpart of NVIDIA's cuDNN). Setting the Find Mode to `2`, or `FAST` is crucial for optimal pretraining speed, otherwise pretraining is 10-20x times slower. More details on MIOpen Find modes are available [here](https://rocm.docs.amd.com/projects/MIOpen/en/docs-5.6.0/find_and_immediate.html) 

* You can simulate 128 GPUs by using k GPUs and adding command line parameters (before `--config-dir`)
`distributed_training.distributed_world_size=k` `+optimization.update_freq='[x]'` where x = 128/k

* For more LUMI-specific details on training with AMD GPUs, see [here](https://lumi-supercomputer.github.io/LUMI-training-materials/4day-20231003/extra_4_10_Best_Practices_GPU_Optimization/), [here](https://lumi-supercomputer.github.io/LUMI-EasyBuild-docs/p/PyTorch/), and [here](https://462000265.lumidata.eu/ai-20240529/files/LUMI-ai-20240529-10-Extreme_scale_AI.pdf).

## Fine-tuning the models with CTC using 🤗Transformers

To fine-tune a wav2vec 2.0 Base model using Hugging Face Transformers, run

```
sbatch scripts/finetuning/low-resource-asr/finetune_wav2vec2_base.sh
```

# Citation

If you use our models or scripts, please cite our work as:

```bibtex
@misc{getman202621mhoursrawbroadcast,
      title={{From 2.1M Hours of Raw Broadcast Audio to Finnish Speech SSL: Confidence Filtering and Architectural Analysis at Scale}, 
      author={Yaroslav Getman and Tamás Grósz and Tommi Lehtonen and Mikko Kurimo},
      year={2026}},
      eprint={2610.13102},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2610.13102}, 
}

@inproceedings{getman25_interspeech,
  title     = {{Is your model big enough? Training and interpreting large-scale monolingual speech foundation models},
  author    = {Yaroslav Getman and Tamás Grósz and Tommi Lehtonen and Mikko Kurimo},
  year      = {2025},
  booktitle = {Interspeech 2025},
  pages     = {231--235},
  doi       = {10.21437/Interspeech.2025-46},
  issn      = {2958-1796},
}
```
