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

# Qwen trivia
# python3 ./attn_logit_lens/run_attn_logit_lens.py \
#   --model_name_or_path Qwen/Qwen2.5-32B-Instruct \
#   --input_h5 ./process_generations/processed_generations_more_h5/qwen_32B/23_32B_1000_train/balanced/train_verbalised_embeddings.h5 \
#   --n_examples 3 \
#   --top_k 8 \
#   --expected_guess_tokens 2 \
#   --expected_probability_tokens 5 \
#   --low_conf_threshold 0.2 \
#   --high_conf_threshold 0.8 \
#   --device cuda:0 \
#   --gain \
#   --ultimate

# Mistral trivia
# python3 ./attn_logit_lens/run_attn_logit_lens.py \
#   --model_name_or_path mistralai/Mistral-7B-Instruct-v0.1 \
#   --input_h5 ./process_generations/processed_generations_more_h5/mistral/2_200_9_prob_toks/train_verbalised_embeddings.h5 \
#   --n_examples 3 \
#   --top_k 8 \
#   --expected_guess_tokens 5 \
#   --expected_probability_tokens 7 \
#   --extend_probability_span \
#   --low_conf_threshold 0.1 \
#   --high_conf_threshold 0.9 \
#   --device cuda:0 \
#   --gain \
#   --ultimate

# Gemma trivia
python3 ./attn_logit_lens/run_attn_logit_lens.py \
  --model_name_or_path google/gemma-3-12b-it \
  --input_h5 ./process_generations/processed_generations_more_h5/new_gemma/4_trivia_gemma_extended_with_concat_3000_train/balanced/train_verbalised_embeddings.h5 \
  --n_examples 3 \
  --top_k 8 \
  --expected_guess_tokens 3 \
  --expected_probability_tokens 5 \
  --extend_probability_span \
  --low_conf_threshold 0.2 \
  --high_conf_threshold 0.8 \
  --device cuda:0 \
  --gain \
  --ultimate
