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
conda activate se_probes
source /vol/cuda/13.0.0/setup.sh
/usr/bin/nvidia-smi
uptime

# Example with explicit layer-head selection:
# python3 ./headwise_mass_mean_probe/run_headwise_mass_mean_probe.py \
#   --input_h5 ./process_generations/processed_generations_more_h5/2_200_concat/train_verbalised_embeddings.h5 \
#   --no-enable_brief \
#   --num_samples 200 \
#   --device cuda:0 \
#   --ablate_heads_by_layer 12.3,12.7,13.2,14.5 \
#   --ablation_mode none probability_tokens_mean_replace \
#   --alpha -1 0.0 0.5 1.0 2.0 \
#   --ablation_targets low high

python3 ./headwise_mass_mean_probe/run_headwise_mass_mean_probe.py \
  --input_h5 ./process_generations/processed_generations_more_h5/2_200_concat/train_verbalised_embeddings.h5 \
  --no-enable_brief \
  --num_samples 200 \
  --device cuda:0 \
  --ablate_heads_by_layer 14.26,10.28,11.11,15.28,16.10,12.31,15.27,10.29,12.22,10.25,13.5,12.14,14.29,11.7,13.22,10.13,10.19,11.1,10.15,13.6,12.11,11.8,13.21 \
  --ablation_mode none probability_tokens_mean_replace \
  --alpha 0.0 1.0 2.0 \
  --ablation_targets low high
