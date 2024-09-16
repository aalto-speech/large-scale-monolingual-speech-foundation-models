from pickle import load
import glob
import os
import argparse
import tarfile
import numpy as np
import pandas as pd
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument('path_to_manifests')
parser.add_argument('channel')
parser.add_argument('valid_size_hours')
args = parser.parse_args()

sr = 16000
target_valid_size_nsamples = sr * int(args.valid_size_hours) * 3600
valid_size_nsamples = 0
df_valid = pd.DataFrame()

path_to_train_manifest = f"{args.path_to_manifests}/{args.channel}/train_and_valid.tsv"
df_train = pd.read_csv(
    path_to_train_manifest,
    delimiter = "\t",
    header = None
)

print(f"{len(df_train)} train samples before sampling validation samples")

while valid_size_nsamples < target_valid_size_nsamples:
    valid_sample = df_train.sample(n=1)
    df_valid = pd.concat([df_valid,valid_sample]).reset_index(drop=True)
    valid_size_nsamples += int(valid_sample[1].item())
    df_train = df_train.loc[~df_train.index.isin(valid_sample.index)].reset_index(drop=True)

print(f"{len(df_train)} train samples ({sum(df_train[1]/sr/3600)} hours) after sampling validation samples")
print(f"{len(df_valid)} validation samples ({sum(df_valid[1]/sr/3600)} hours)")

df_train.to_csv(
    f"{args.path_to_manifests}/{args.channel}/train.tsv",
    sep="\t",
    index=False,
    header=None
    )

df_valid.to_csv(
    f"{args.path_to_manifests}/{args.channel}/valid.tsv",
    sep="\t",
    index=False,
    header=None
    )
