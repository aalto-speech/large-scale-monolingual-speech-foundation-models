import glob, os
import numpy as np
import librosa
import soundfile as sf
import torch
from pyannote.audio import Pipeline
from pathlib import Path
from pickle import load, dump
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('year')
parser.add_argument('month')
args = parser.parse_args()
sr = 16000
max_segment_dur_secs = 30
max_segment_length = sr * max_segment_dur_secs
path_to_tars = "/radio_station_1_tar"
path_to_flac = "/radio_station_1_flac"

start=time.time()

audio_format = ".flac"
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
t='my_huggingface_access_token'
pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection",use_auth_token=t)
pipeline.to(device)
# print(pipeline.parameters(instantiated=True))
# {'min_duration_off': 0.09791355693027545, 'min_duration_on': 0.05537587440407595,
#  'offset': 0.4806866463041527, 'onset': 0.8104268538848918}

initial_params = {"onset": 0.8104268538848918, "offset": 0.4806866463041527,
                "min_duration_on": 2.0, "min_duration_off": 0.09791355693027545}
pipeline.instantiate(initial_params)

try:
    with open(f"{path_to_tars}/fname_to_nframes_dict_radio_station_1_{args.year}.pickle", "rb") as f:
        out_file_to_nframes_dict = load(f)
except FileNotFoundError:
    out_file_to_nframes_dict = {}

count_segmented=0
total_dur=0
for file in glob.glob(f"{path_to_flac}/{args.year}/{args.month}/**/*{audio_format}",recursive=True):
    try:
        y,sr=librosa.load(file, sr=16000)
        output = pipeline({"waveform": torch.unsqueeze(torch.tensor(y), 0), "sample_rate": sr})
        segments_list = output.get_timeline().segments_list_
        if len(segments_list) > 0:
            count_segments=0
            total_dur+=len(y)
            count_segmented+=1
            # for idx, item in enumerate(segments_list):
            for item in segments_list:
                if item.end-item.start>30:
                    long_segment = y[int(item.start*sr):int(item.end*sr)]
                    num_segments = int(np.ceil(len(long_segment) / max_segment_length))
                    for i in range(num_segments):
                        segment = long_segment[i * max_segment_length: (i + 1) * max_segment_length]
                        if len(segment)>=32000:
                            out_file = file.replace("/radio_station_1_flac/","/radio_station_1_flac_segmented/").replace(".flac",f"_part_{count_segments}.flac")
                            Path(out_file).parent.mkdir(parents=True, exist_ok=True)
                            sf.write(out_file, segment, sr)
                            out_file_to_nframes_dict[out_file.replace("/radio_station_1_flac_segmented/","")]=len(segment)
                            count_segments+=1
                else:
                    segment = y[int(item.start*sr):int(item.end*sr)]
                    out_file = file.replace("/radio_station_1_flac/","/radio_station_1_flac_segmented/").replace(".flac",f"_part_{count_segments}.flac")
                    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
                    sf.write(out_file, segment, sr)
                    out_file_to_nframes_dict[out_file.replace("/radio_station_1_flac_segmented/","")]=len(segment)
                    count_segments+=1
    except Exception as error:
        print(f"{type(error).__name__} occurred when opening file {file} ({error})")

l = len(glob.glob(f"{path_to_flac}/{args.year}/{args.month}/**/*{audio_format}",recursive=True))
end=time.time()
print(f"Ylepuhe {args.year} {args.month}")
print(f"Segmented {count_segmented} out of {l} files ({int(total_dur/sr/3600)}h) in {(end-start)/3600}h")
with open(f"{path_to_tars}/fname_to_nframes_dict_radio_station_1_{args.year}.pickle", "wb") as f:
    dump(out_file_to_nframes_dict, f)
