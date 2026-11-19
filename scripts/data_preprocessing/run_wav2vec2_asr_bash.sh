path_to_flac=/flac_original
path_to_tars=/tar_vad_only/all
path_to_scripts=/scripts/data_preprocessing
path_to_input_manifests=/tsv_no_headers_vad_only/train_and_valid
path_to_output_manifests=/tsv_vad_only_with_asr/train_and_valid

tv_radio=tv

for station in  yletv1 yletv2; do
  mkdir -p ${path_to_scripts}/logs/${tv_radio}/${station}
  mkdir -p ${path_to_output_manifests}/${tv_radio}

  sbatch --job-name=asr_${station} \
         --output=${path_to_scripts}/logs/${tv_radio}/${station}/run_wav2vec2_asr_${station}.o \
         --error=${path_to_scripts}/logs/${tv_radio}/${station}/run_wav2vec2_asr_${station}.e \
         --export=path_to_flac=${path_to_flac},path_to_tars=${path_to_tars},path_to_scripts=${path_to_scripts},tv_radio=${tv_radio},station=${station},path_to_input_manifests=$path_to_input_manifests,path_to_output_manifests=${path_to_output_manifests} \
         ${path_to_scripts}/run_wav2vec2_asr.sh
done
