for layernum in {1..12}
do
sbatch --job-name=IG_${layernum} --output=/ig_layer_${layernum}.out --export=layernum=${layernum} scripts/interpretation/ig_single_layer.sh
done
