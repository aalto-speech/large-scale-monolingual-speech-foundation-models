from transformers import Wav2Vec2ForCTC, AutoProcessor
import datasets
import re
import string
import time
import torch
import argparse
from captum.attr import LayerIntegratedGradients

from transformers import Wav2Vec2ForCTC, AutoProcessor
import datasets
import re
import string

def forward(input):
    return model(input).logits[0]

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--layer_num", type=int, help='Transformer layer number')
args = parser.parse_args()

cache_dir="/cache"
test_dataset = datasets.load_dataset("google/fleurs", "fi_fi", split='test', cache_dir=cache_dir)
test_dataset = test_dataset.rename_column("path", "file")
test_dataset = test_dataset.rename_column("transcription", "sentence")
test_dataset = test_dataset.rename_column("audio", "speech")

def prepare_example_fleurs(example):
    example["audio"] = example["speech"]["array"]
    example["sampling_rate"] = example["speech"]["sampling_rate"]
    example["duration_in_seconds"] = len(example["audio"]) / example["sampling_rate"]
    transcription = example["sentence"]
    transcription = transcription.translate(str.maketrans('', '', string.punctuation.replace("'","")))
    transcription = re.sub(' +', ' ', transcription).lower()
    example["text"] = transcription
    return example
test_dataset = test_dataset.map(prepare_example_fleurs, remove_columns=["speech"])

model_path_large_ft = "/wav2vec2_base_finetuned"
batch_size = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Wav2Vec2ForCTC.from_pretrained(
        model_path_large_ft,
        cache_dir=cache_dir
).to(device)
processor = AutoProcessor.from_pretrained(
    model_path_large_ft,
    cache_dir=cache_dir
)

internal_batch_size = 2
attribution_final_list = []
intermediate_results_saved = False
predicted_ids_list = []
start_time = time.time()
for item_idx, item in enumerate(test_dataset):
    input_values = processor(item["audio"], sampling_rate=16000, return_tensors="pt", padding="longest").input_values.to(device)
    logits = model(input_values.to(device)).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    unique_predicted_ids = torch.unique(predicted_ids)

    for target in unique_predicted_ids:
        predicted_id_indices = torch.where(predicted_ids[0]==target.item())[0]
        layer_ig = LayerIntegratedGradients(forward, model.wav2vec2.encoder.layers[args.layer_num-13], multiply_by_inputs=True)

        attribution = layer_ig.attribute(
            input_values,
            target=target.item(),
            internal_batch_size=internal_batch_size,
            attribute_to_layer_input=False
        )

        attribution_filtered = torch.stack([attribution[0, predicted_id_indices[idx].item()] for idx in range(predicted_id_indices.shape[0])], dim=0)
        attribution_final_list += attribution_filtered
        predicted_ids_list += [target.item()]*predicted_id_indices.shape[0]

attribution_final = torch.stack(attribution_final_list, dim=0)
elapsed_time = time.time() - start_time
print(f"Elapsed time: {elapsed_time/3600}h")
torch.save(attribution_final, f"/attr_fleurs_L{args.layer_num}.pt")
if args.layer_num == 1:
    torch.save(torch.tensor(predicted_ids_list), f"/predicted_ids_fleurs.pt")
