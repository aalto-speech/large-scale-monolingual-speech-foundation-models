import pandas as pd
import os, glob

input_tsv_base_dir = "/tsv_vad_only_with_asr/train_and_valid"
output_tsv_base_dir = "/tsv_vad_only_with_asr_deduplicated_nonempty/train_and_valid"

for tsv in glob.glob(f"{input_tsv_base_dir}/**/*_train_and_valid.tsv", recursive=True):
    df = pd.read_csv(tsv, sep='\t')

    total_hours_unfiltered = df['nframes'].sum() / 16000 / 3600

    df = df[~df.transcript.isna()]
    total_hours_wo_empty_transcripts = df['nframes'].sum() / 16000 / 3600

    df = df.sort_values('non_blank_log_prob_mean', ascending=False)
    df = df.drop_duplicates(subset='transcript', keep='first')
    total_hours_filtered = df['nframes'].sum() / 16000 / 3600

    station = os.path.basename(tsv).replace("_train_and_valid.tsv", "")

    print(f"{station}: {total_hours_unfiltered:.2f}/{total_hours_wo_empty_transcripts:.2f}/{total_hours_filtered:.2f} "
          f"before/after removing empty transcripts and duplicates")

    out_tsv = tsv.replace(input_tsv_base_dir, output_tsv_base_dir)
    os.makedirs(os.path.dirname(out_tsv), exist_ok=True)

    df.to_csv(out_tsv, index=False, sep='\t')
