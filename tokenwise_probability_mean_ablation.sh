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

# Standard tokenwise run across a layer span:
python3 ./tokenwise_probability_mean_ablation/run_tokenwise_probability_mean_ablation.py \
  --input_h5 ./process_generations/processed_generations_more_h5/2_200_concat/train_verbalised_embeddings.h5 \
  --no-enable_brief \
  --num_samples 80 \
  --device cuda:0 \
  --ablate_layers 10-16 \
  --new_h5_format \
  --individual_layers

# Individual-layer mode (uncomment to run layer x token grid output):
# python3 ./tokenwise_probability_mean_ablation/run_tokenwise_probability_mean_ablation.py \
#   --input_h5 ./process_generations/processed_generations_more_h5/1_1000_prob_val_span/train_verbalised_embeddings.h5 \
#   --no-enable_brief \
#   --num_samples 1000 \
#   --device cuda:0 \
#   --new_h5_format \
#   --no-mean_from_low_confidence \
#   --individual_layers
