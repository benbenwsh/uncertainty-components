#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=${USER} # required to send email notifcations - alternatively, enter an email address
#SBATCH --output=./slurm_out/slurm-%j.out
export PATH=/vol/bitbucket/${USER}/myvenv/bin/:$PATH
# for CephFS - remember to follow Step 2 to create folders
# /vol/gpudata/path-to-folder/myvenv/bin/:$PATH
# the above path could also point to a miniconda install
# if using miniconda, uncomment the below line
source ~/.bashrc
source activate
conda activate uncertainty_components
source /vol/cuda/13.0.0/setup.sh
/usr/bin/nvidia-smi
uptime

# for layer in $(seq 0 31); do
#   python3 ./headwise_mean_ablation/run_headwise_mean_ablation.py --input_h5 ./process_generations/processed_generations_more_h5/2_200_concat/train_verbalised_embeddings.h5 --num_samples 200 --device cuda:0 --ablate_layers "${layer}" --no-mean_from_low_confidence
# done

# Example with explicit layer-head selection:
# python3 ./headwise_mean_ablation/run_headwise_mean_ablation.py \
#   --input_h5 ./process_generations/processed_generations_more_h5/2_200_concat/train_verbalised_embeddings.h5 \
#   --no-enable_brief \
#   --num_samples 200 \
#   --device cuda:0 \
#   --ablate_heads_by_layer 12.3,12.7,13.2,14.5 \
#   --no-mean_from_low_confidence

python3 ./headwise_mean_ablation/run_headwise_mean_ablation.py \
  --input_h5 ./process_generations/processed_generations_more_h5/2_200_concat/train_verbalised_embeddings.h5 \
  --no-enable_brief \
  --num_samples 200 \
  --device cuda:0 \
  --ablate_heads_by_layer 10.7,10.8,10.31,11.9,11.24,11.25,12.3,12.14,12.29,13.3,13.20,13.24,14.9,14.11,14.13,15.2,15.6,15.7,16.10,16.25,16.28 \
  --ablation_mode none probability_tokens_mean_replace \
  --no-mean_from_low_confidence



