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

# Tokenwise direction steering across all layers (layer x token grid):
python3 ./tokenwise_probability_mass_mean_probe/run_tokenwise_probability_mass_mean_probe.py \
  --model_name mistralai/Mistral-7B-Instruct-v0.1 \
  --input_h5 ./process_generations/processed_generations_more_h5/mistral/2_200_9_prob_toks/train_verbalised_embeddings.h5 \
  --dataset trivia_qa \
  --no-enable_brief \
  --num_samples 22 \
  --device cuda:0 \
  --dtype bfloat16 \
  --new_h5_format \
  --individual_layers \
  --alpha 1.0 \
  --expected_guess_tokens 5 \
  --expected_probability_tokens 7 \
  --extend_probability_span \
  --low_conf_threshold 0.1 \
  --high_conf_threshold 0.9 \
  --no-mean_from_low_confidence

# Low-confidence cohort (uncomment to steer low-conf examples toward higher confidence):
# python3 ./tokenwise_probability_mass_mean_probe/run_tokenwise_probability_mass_mean_probe.py \
#   --input_h5 ./process_generations/processed_generations_more_h5/2_200_concat/train_verbalised_embeddings.h5 \
#   --no-enable_brief \
#   --num_samples 80 \
#   --device cuda:0 \
#   --new_h5_format \
#   --individual_layers \
#   --alpha 1.0 \
#   --no-mean_from_low_confidence
