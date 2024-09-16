from pickle import load
import glob
import os
import argparse
import tarfile
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('path_to_manifests')
parser.add_argument('channel')
args = parser.parse_args()
audio_format = ".flac"
if os.path.exists(f"{args.path_to_manifests}/{args.channel}/train_and_valid.tsv"):
    os.remove(f"{args.path_to_manifests}/{args.channel}/train_and_valid.tsv")
train_tsv = open(f"{args.path_to_manifests}/{args.channel}/train_and_valid.tsv", 'w')
nframes_total=0

for idx,tarfname in enumerate(sorted(glob.glob(f"/{args.channel}_tar/*.tar"))):
    year = Path(tarfname).stem.split('_')[-1]
    pickle_path = f"{os.path.dirname(tarfname)}/fname_to_nframes_dict_{os.path.basename(tarfname).replace('.tar','.pickle')}"
    with open(pickle_path, "rb") as g:
        fname_to_nframes_dict = load(g)
    if len(next(os.walk(f"/{args.channel}_flac/{year}"))[1]) != len(np.unique(np.array([item.split('/')[1] for item in  list(fname_to_nframes_dict.keys())]))):
        print(f"{args.channel} Year {year}: INCORRECT NUMBER OF SUB-FOLDERS!!!")
    with tarfile.open(tarfname, 'r') as tar:
        line_sep = "" if idx==0 else "\n"
        for tarinfo in tar:
            if audio_format in tarinfo.name:
                try:
                    nframes=fname_to_nframes_dict[tarinfo.name]
                except KeyError:
                    nframes=fname_to_nframes_dict[tarinfo.name.replace(".aac_","_")]
                line = f"{line_sep}{Path(tarfname).name}:{tarinfo.offset_data}:{tarinfo.size}\t{nframes}"
                train_tsv.write(line)
                line_sep = "\n"
                nframes_total+=nframes
train_tsv.close()

print(f"{args.channel}: {idx+1} tar archives preprocessed and stored to the manifest")
print(f"{args.channel}: {nframes_total/16000/3600} hours after VAD preprocessing")
