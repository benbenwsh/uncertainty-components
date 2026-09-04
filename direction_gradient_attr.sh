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

# python3 ./direction_gradient_attr/run_direction_gradient_attr.py \
#   --model_name google/gemma-3-12b-it \
#   --input_h5 ./process_generations/processed_generations_more_h5/new_gemma/4_trivia_gemma_extended_with_concat_3000_train/balanced/train_verbalised_embeddings.h5 \
#   --device cuda:0 \
#   --dtype bfloat16 \
#   --expected_probability_tokens 5 \
#   --max_examples_for_mean 50 \
#   --granularity coarse fine

# Mistral version (works)
# python3 ./direction_gradient_attr/run_direction_gradient_attr.py \
#   --model_name mistralai/Mistral-7B-Instruct-v0.1 \
#   --input_h5 ./process_generations/processed_generations_more_h5/mistral/2_200_9_prob_toks/train_verbalised_embeddings.h5 \
#   --device cuda:0 \
#   --dtype bfloat16 \
#   --expected_probability_tokens 7 \
#   --max_examples_for_mean 50 \
#   --granularity coarse fine

# Default --rerun_autoregressive greedy-decodes Guess:/Probability: from each H5 question.
# Use --no-rerun_autoregressive to reconstruct the prefix from stored decoded_tokens.

# Linguistic Confidence (Mistral only):
python3 ./direction_gradient_attr/run_direction_gradient_attr.py \
  --model_name mistralai/Mistral-7B-Instruct-v0.1 \
  --input_h5 ./process_generations/processed_generations_more_h5/3_trivia_mistral_linguistic_train/balanced/train_verbalised_embeddings.h5 \
  --device cuda:0 \
  --dtype bfloat16 \
  --linguistic_confidence_prompt \
  --expected_confidence_tokens 5 \
  --high_conf_threshold 0.9 \
  --low_conf_threshold 0.1 \
  --max_examples_for_mean 50 \
  --granularity coarse fine
