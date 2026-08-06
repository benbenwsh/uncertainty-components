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

# python3 ./tokenwise_probability_mean_ablation/run_tokenwise_probability_mean_ablation.py \
#   --model_name Qwen/Qwen2.5-32B-Instruct \
#   --input_h5 ./process_generations/processed_generations_more_h5/qwen_32B/22_svamp_32B_train/train_verbalised_embeddings.h5 \
#   --dataset svamp \
#   --no-enable_brief \
#   --num_samples 50 \
#   --device cuda:0 \
#   --dtype bfloat16 \
#   --new_h5_format \
#   --low_conf_threshold 0.1 \
#   --high_conf_threshold 0.9 \
#   --individual_layers \
#   --no-mean_from_low_confidence

# Individual-layer mode (uncomment to run layer x token grid output):
# # Qwen
# python3 ./tokenwise_probability_mean_ablation/run_tokenwise_probability_mean_ablation.py \
#   --model_name Qwen/Qwen2.5-32B-Instruct \
#   --input_h5 ./process_generations/processed_generations_more_h5/qwen_32B/23_32B_1000_train/train_verbalised_embeddings.h5 \
#   --dataset trivia_qa \
#   --no-enable_brief \
#   --num_samples 50 \
#   --device cuda:0 \
#   --dtype bfloat16 \
#   --new_h5_format \
#   --low_conf_threshold 0.25 \
#   --high_conf_threshold 0.9 \
#   --individual_layers \

# Gemma
python3 ./tokenwise_probability_mean_ablation/run_tokenwise_probability_mean_ablation.py \
  --model_name Qwen/Qwen2.5-32B-Instruct \
  --input_h5 ./process_generations/processed_generations_more_h5/qwen_32B/23_32B_1000_train/balanced/train_verbalised_embeddings.h5 \
  --dataset trivia_qa \
  --no-enable_brief \
  --num_samples 20 \
  --device cuda:0 \
  --dtype bfloat16 \
  --new_h5_format \
  --low_conf_threshold 0.2 \
  --high_conf_threshold 0.8 \
  --individual_layers \
  --expected_guess_tokens 2 \
  --expected_probability_tokens 5 \
  --no-mean_from_low_confidence

# so the oom on cpu is indeed because of the process weight float upcasting
# N.B. num_samples refer to the max number of iterations to perform, not limiting
# the mean vector computation