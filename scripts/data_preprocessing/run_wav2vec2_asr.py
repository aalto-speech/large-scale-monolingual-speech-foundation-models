import os
import io
import re
import tarfile
import mmap
import time
import datetime
import argparse
import logging
from shutil import copyfile
import hostlist
from datasets.utils.logging import enable_progress_bar, disable_progress_bar
import re
import shutil

import numpy as np
import pandas as pd
import soundfile as sf

import torch
import torch.nn.functional as F
import torchaudio
from torch.distributed import (
    init_process_group,
    get_rank,
    get_world_size,
    destroy_process_group,
)

import datasets
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def get_logger():
    return logging.getLogger()


def main():
    parser = argparse.ArgumentParser(description="Distributed ASR with wav2vec2")
    parser.add_argument("--input_tsv", type=str, default="/scratch/project_462000971/Fin_w2v2_data_full_042024/tsv_project_462000971/valid_FULL.tsv")
    parser.add_argument("--input_tsv_header", type=str, default=None)
    parser.add_argument("--output_tsv", type=str, default="/scratch/project_462000971/Fin_w2v2_data_full_072025/data_preprocessing_scripts/vad_and_lid/asr_only/data_splits_after_asr/valid_FULL.tsv")
    parser.add_argument("--model_name", type=str, default="/scratch/project_462000971/home/getmanya/icassp_2024_best_models/wav2vec2_large/FT/wav2vec2_large_best_chkpt_2_ft_HF", help="Wav2Vec2 model name or path")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per GPU")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with more logging")
    parser.add_argument("--tv_radio", type=str, default="tv", help="tv/radio/MEDIA/previous_project")
    parser.add_argument("--station", type=str, default="alfatv", help="station")
    parser.add_argument("--shared_storage", type=str, default="/scratch/project_462000857/shared_temp", help="Shared storage path accessible by all nodes")
    parser.add_argument("--local_temp", type=str, default="/tmp", help="Local temp directory for each node")
    args = parser.parse_args()

    try:
        global_rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_rank = int(os.environ["SLURM_LOCALID"])
        node_id = int(os.environ["SLURM_NODEID"])
        node_list = os.environ.get("SLURM_STEP_NODELIST")
        if node_list is None:
            node_list = os.environ.get("SLURM_JOB_NODELIST")
        hostnames = hostlist.expand_hostlist(node_list)
        is_distributed = True

        init_process_group(
            backend="nccl",
            rank=global_rank,
            world_size=world_size,
            init_method=f"tcp://{hostnames[0]}:29500",
            timeout=datetime.timedelta(minutes=16000)
        )
        
    except (KeyError, ValueError) as e:
        print(f"SLURM environment not found, falling back to single process: {e}")
        global_rank = 0
        world_size = 1
        local_rank = 0
        node_id = 0
        is_distributed = False

    logging.basicConfig(
        level=logging.INFO if not args.debug else logging.DEBUG,
        format=f"[Rank {global_rank}] %(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger()

    logger.info(f"Process {global_rank}/{world_size} initialized (node {node_id}, local rank {local_rank})")

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"
    logger.info(f"Using device: {device}")

    if global_rank == 0:
        os.makedirs(args.shared_storage, exist_ok=True)
        timestamp = int(time.time())
        with open(f"{args.shared_storage}/run_timestamp.txt", "w") as f:
            f.write(str(timestamp))
    
    # Wait for rank 0 to create shared storage
    if is_distributed:
        torch.distributed.barrier()

    # Timestamp for unique file names
    with open(f"{args.shared_storage}/run_timestamp.txt") as f:
        timestamp = int(f.read().strip())


    def load_fairseq_dataset(data_files, rank, timestamp, script_cache_dir, header=None):
        logger = logging.getLogger()
        logger.info(f"Loading dataset from {data_files}")

        # Read the first line to get the root path
        if not header:
            with open(data_files, 'r') as f:
                root_path = f.readline().strip()
        else:
            root_path = header
        logger.info(f"Root path: {root_path}")

        # Parse remaining lines
        data_rows = []
        with open(data_files, 'r') as f:
            if not header:
                f.readline() # skip the root path line
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) != 2:
                    logger.warning(f"Malformed line: {line}")
                    continue
                file_part, nframes = parts
                file_components = file_part.split(':')
                if len(file_components) != 3:
                    logger.warning(f"Malformed file part: {file_part}")
                    continue
                filename, offset, length = file_components
                data_rows.append({
                    "file": f"{root_path}/{filename}",
                    "offset": int(offset),
                    "length": int(length),
                    "nframes": int(nframes)
                })

        logger.info(f"Parsed {len(data_rows)} data rows")

        df = pd.DataFrame(data_rows)
        temp_csv = f"/tmp/temp_dataframe_rank{rank}_{timestamp}.csv"
        logger.info(f"Saving temporary CSV to {temp_csv}")
        df.to_csv(temp_csv, encoding="utf-8", index=False)

        logger.info("Loading dataset from temporary CSV")

        
        ds = datasets.load_dataset("csv", data_files=temp_csv, split="train", cache_dir=script_cache_dir)
        # ds = datasets.load_dataset("csv", data_files=temp_csv, split="train", cache_dir="/tmp")

        logger.info(f"Loaded dataset with {len(ds)} examples")

        os.remove(temp_csv)

        return ds


    def prepare_example_from_tar(example):
        logger = logging.getLogger()
        try:
            with open(example["file"], "rb") as f:
                with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mmap_o:
                    data = mmap_o[example["offset"] : example["offset"] + example["length"]]
            f = io.BytesIO(data)
            example["audio"], _ = sf.read(f, dtype="float32")
            return example
        except Exception as e:
            logger.error(
                f"Error preparing example from tar: {e}, file: {example.get('file','unknown')}, "
                f"offset: {example.get('offset','unknown')}"
            )
            return None


    def compute_ctc_stats(logits, pred_ids, blank_id):
        log_probs = F.log_softmax(logits, dim=-1)  # [B, T, V]
        pred_logprobs = torch.gather(
            log_probs, dim=-1, index=pred_ids.unsqueeze(-1)
        ).squeeze(-1)  # [B, T]

        non_blank_mask = (pred_ids != blank_id).float()  # [B, T]
        blank_mask = 1.0 - non_blank_mask                # [B, T]
        counts = torch.stack([
            non_blank_mask.sum(dim=1),  # [B]
            blank_mask.sum(dim=1)       # [B]
        ], dim=1)  # [B, 2]

        batch_size, seq_len = logits.shape[:2]
        device = logits.device

        sums = torch.stack([
            (pred_logprobs * non_blank_mask).sum(dim=1),
            (pred_logprobs * blank_mask).sum(dim=1)
        ], dim=1)  # [B, 2]

        # NaN if count=0
        means = torch.where(
            counts > 0,
            sums / counts,
            torch.full_like(sums, float("nan"))
        )  # [B, 2]

        # Compute squared diffs
        diffs_non_blank = (pred_logprobs - means[:, 0].unsqueeze(1)) * non_blank_mask
        diffs_blank = (pred_logprobs - means[:, 1].unsqueeze(1)) * blank_mask
        squared_sums = torch.stack([
            diffs_non_blank.pow(2).sum(dim=1),
            diffs_blank.pow(2).sum(dim=1)
        ], dim=1)  # [B, 2]

        # NaN if count=0, 0 if count=1
        stds = torch.where(
            counts > 1,
            torch.sqrt(squared_sums / (counts - 1)),
            torch.full_like(squared_sums, float("nan"))
        )
        stds = torch.where(counts == 1, torch.zeros_like(stds), stds)

        full_mean = pred_logprobs.mean(dim=1)
        full_std = (
            pred_logprobs.std(dim=1, unbiased=True)
            if seq_len > 1
            else torch.zeros(batch_size, device=device)
        )

        non_blank_mean, blank_mean = means[:, 0], means[:, 1]
        non_blank_std, blank_std = stds[:, 0], stds[:, 1]

        return non_blank_mean, non_blank_std, blank_mean, blank_std, full_mean, full_std


    def map_to_pred(batch):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        inputs = processor(
            batch["audio"],
            sampling_rate=16000,
            return_tensors="pt",
            padding="longest"
            )
        input_values = inputs.input_values.to(device)

        attention_mask = getattr(inputs, 'attention_mask', None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        with torch.no_grad():
            outputs = model(input_values, attention_mask=attention_mask) if attention_mask is not None else model(input_values)
            
        pred_ids = torch.argmax(outputs.logits, dim=-1)

        non_blank_mean, non_blank_std, blank_mean, blank_std, full_mean, full_std = compute_ctc_stats(
            outputs.logits, pred_ids, processor.tokenizer.pad_token_id
        )

        batch["transcription"] = processor.batch_decode(pred_ids)
        batch["non_blank_log_prob_mean"] = non_blank_mean.cpu().tolist()
        batch["non_blank_log_prob_std"] = non_blank_std.cpu().tolist()
        batch["blank_log_prob_mean"] = blank_mean.cpu().tolist()
        batch["blank_log_prob_std"] = blank_std.cpu().tolist()
        batch["full_log_prob_mean"] = full_mean.cpu().tolist()
        batch["full_log_prob_std"] = full_std.cpu().tolist()

        return batch


    # Split work across ranks
    base_input_name = os.path.basename(args.output_tsv)
    base_input_name_without_ext = os.path.splitext(base_input_name)[0]
    split_dir = os.path.join(args.shared_storage, f"splits_{timestamp}")
    script_cache_dir = f"{args.shared_storage}/cache/{args.tv_radio}_{args.station}_{timestamp}"
    if global_rank == 0:
        os.makedirs(split_dir, exist_ok=True)
        shard_prefix = os.path.join(split_dir, f"{base_input_name_without_ext}_")
        cmd = (
            f"split -n l/{world_size} "
            f"--numeric-suffixes=0 --suffix-length=3 "
            f"--additional-suffix=.tsv {args.input_tsv} {shard_prefix}"
        )
        logger.info(f"Splitting TSV into {world_size} shards: {cmd}")
        os.system(cmd)

        os.makedirs(script_cache_dir, exist_ok=True)

    if is_distributed:
        torch.distributed.barrier()

    shard_file = os.path.join(split_dir, f"{base_input_name_without_ext}_{global_rank:03d}.tsv")
    logger.info(f"Rank {global_rank} loading {shard_file}")

    dataset = load_fairseq_dataset(shard_file, global_rank, timestamp, script_cache_dir, args.input_tsv_header)
    os.remove(shard_file)

    if is_distributed:
        logger.info(
            f"Rank {global_rank}/{world_size} processing {len(dataset)} examples "
        )
        disable_progress_bar()
        dataset = dataset.map(
            prepare_example_from_tar,
            # num_proc=7
        )

    dataset = dataset.sort("nframes")

    logger.info(f"Loading wav2vec2 model {args.model_name} on device {device}")
    processor = Wav2Vec2Processor.from_pretrained(args.model_name, cache_dir="/scratch/project_462000971/home/getmanya/cache")
    model = Wav2Vec2ForCTC.from_pretrained(args.model_name, cache_dir="/scratch/project_462000971/home/getmanya/cache").to(device)
    model.eval()

    base_output_name = os.path.basename(args.output_tsv)
    base_name_without_ext = os.path.splitext(base_output_name)[0]
    temp_output = f"{args.shared_storage}/{base_name_without_ext}_rank_{global_rank}_{timestamp}.csv"

    # enable_progress_bar()

    logger.info(f"Rank {global_rank}: Running batched inference with batch size {args.batch_size}")
    process_start_time = time.time() 
    dataset = dataset.map(
        map_to_pred,
        batched=True,
        batch_size=args.batch_size,
        remove_columns=["audio"],
    )

    # rank_csv = f"{args.shared_storage}/{base_name_without_ext}_rank_{global_rank}_{timestamp}.csv"
    
    dataset = dataset.select_columns([
        "file",
        "offset",
        "length",
        "nframes",
        "transcription",
        "non_blank_log_prob_mean",
        "non_blank_log_prob_std",
        "blank_log_prob_mean",
        "blank_log_prob_std",
        "full_log_prob_mean",
        "full_log_prob_std",
    ])
    dataset.to_csv(temp_output, index=False)
    logger.info(f"Saved rank CSV: {temp_output}")

    if is_distributed:
        logger.info(f"Rank {global_rank}: waiting at barrier before combining results")
        torch.distributed.barrier()
        logger.info(f"Rank {global_rank}: passed barrier, proceeding to combine results")

    if (not is_distributed) or (is_distributed and global_rank == 0):
        logger.info("Combining results from all workers")

        output_dir = os.path.dirname(args.output_tsv)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        csv_output = args.output_tsv.replace('.tsv', '_complete_results.csv')
        tsv_output = args.output_tsv

        for out_file in [csv_output, tsv_output]:
            if os.path.exists(out_file):
                try:
                    os.remove(out_file)
                    logger.info(f"Removed existing output file: {out_file}")
                except Exception as e:
                    logger.error(f"Failed to remove existing output file {out_file}: {e}")

        with open(csv_output, 'w', encoding='utf-8') as csv_f, open(tsv_output, 'w', encoding='utf-8') as tsv_f:
            csv_f.write(
                "file,offset,length,nframes,transcript,"
                "non_blank_log_prob_mean,non_blank_log_prob_std,blank_log_prob_mean,blank_log_prob_std,full_log_prob_mean,full_log_prob_std\n"
            )
            tsv_f.write(
                "file\tnframes\ttranscript\tnon_blank_log_prob_mean\tnon_blank_log_prob_std\tblank_log_prob_mean\tblank_log_prob_std\tfull_log_prob_mean\tfull_log_prob_std\n"
            )

            for r in range(world_size):
                rank_csv = f"{args.shared_storage}/{base_name_without_ext}_rank_{r}_{timestamp}.csv"
                logger.info(f"Appending data from {rank_csv}")

                if not os.path.exists(rank_csv):
                    logger.warning(f"Missing rank CSV: {rank_csv}")
                    continue

                with open(rank_csv, 'r', encoding='utf-8') as f:
                    next(f)  # skip header
                    for line in f:
                        csv_f.write(line)

                        parts = line.rstrip('\n').split(',')
                        if len(parts) >= 11:
                            file, offset, length, nframes, transcript, nb_mean, nb_std, b_mean, b_std, f_mean, f_std = parts[:11]
                            nb_mean = 'nan' if not nb_mean and nb_mean != 0.0 else nb_mean
                            nb_std = 'nan' if not nb_std and nb_std != 0.0 else nb_std
                            b_mean = 'nan' if not b_mean and b_mean != 0.0 else b_mean
                            b_std = 'nan' if not b_std and b_std != 0.0 else b_std
                            f_mean = 'nan' if not f_mean and f_mean != 0.0 else f_mean
                            f_std = 'nan' if not f_std and f_std != 0.0 else f_std
                            archive_name = os.path.basename(file)
                            file_id = f"{archive_name}:{offset}:{length}"
                            tsv_f.write(
                                f"{file_id}\t{nframes}\t{transcript}\t{float(nb_mean):.3f}\t{float(nb_std):.3f}"
                                f"\t{float(b_mean):.3f}\t{float(b_std):.3f}\t{float(f_mean):.3f}\t{float(f_std):.3f}\n"
                            )
                logger.info(f"Combined CSV saved to {csv_output}")
                logger.info(f"ASR TSV saved to {tsv_output}")

            for r in range(world_size):
                rank_csv = f"{args.shared_storage}/{base_name_without_ext}_rank_{r}_{timestamp}.csv"
                try:
                    os.remove(rank_csv)
                    logger.info(f"Deleted temporary rank CSV: {rank_csv}")
                except Exception as e:
                    logger.error(f"Failed to delete temporary rank CSV {rank_csv}: {e}")
            # os.remove(f"{args.shared_storage}/run_timestamp.txt")
            if os.path.exists(args.shared_storage):
                shutil.rmtree(args.shared_storage)
                logger.info(f"Deleted shared storage at {args.shared_storage}")
            process_end_time = time.time()
            duration = (process_end_time - process_start_time) / 60
            logger.info(f"Rank {global_rank}: Batched inference + CSV processing completed in {duration:.2f} minutes")

    try:
        if is_distributed:
            destroy_process_group()
            logging.getLogger().info(f"Rank {global_rank}: Destroyed process group")
    except Exception as e:
        logging.getLogger().error(f"Rank {global_rank}: Failed to destroy process group: {e}")

if __name__ == "__main__":
    main()
