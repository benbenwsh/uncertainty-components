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

# python3 ./componentwise_mass_mean_shift/run_componentwise_mass_mean_shift.py \
#   --model_name google/gemma-3-12b-it \
#   --input_h5 ./process_generations/processed_generations_more_h5/new_gemma/4_trivia_gemma_extended_with_concat_3000_train/balanced/train_verbalised_embeddings.h5 \
#   --no-enable_brief \
#   --num_samples 30 \
#   --device cuda:0 \
#   --dtype bfloat16 \
#   --ablate_heads m20,a21,a20,m29,a25,m25,a17,a21,m21,m23 \
#   --dataset trivia_qa \
#   --low_conf_threshold 0.2 \
#   --high_conf_threshold 0.8 \
#   --expected_guess_tokens 3 \
#   --expected_probability_tokens 5 \
#   --alpha 0.5 1.0 2.0 \
#   --intervention_site output \
#   --ablation_mode none last_a_panl_and_pc_mean_replace probability_tokens_mean_replace

# Mistral input experiments
python3 ./componentwise_mass_mean_shift/run_componentwise_mass_mean_shift.py \
  --model_name mistralai/Mistral-7B-Instruct-v0.1 \
  --input_h5 ./process_generations/processed_generations_more_h5/mistral/2_200_9_prob_toks/train_verbalised_embeddings.h5 \
  --no-enable_brief \
  --num_samples 30 \
  --device cuda:0 \
  --dtype bfloat16 \
  --ablate_heads a5.h28,a6.h5,a25.h16,a27.h23,m4 \
  --dataset trivia_qa \
  --low_conf_threshold 0.1 \
  --high_conf_threshold 0.9 \
  --expected_guess_tokens 5 \
  --expected_probability_tokens 7 \
  --alpha 0.5 1.0 2.0 \
  --intervention_site input \
  --ablation_mode none probability_last_token_mean_replace extended_probability_last_token_mean_replace
  # --ablation_mode none last_a_panl_and_pc_mean_replace probability_tokens_mean_replace

# python3 ./componentwise_mass_mean_shift/run_componentwise_mass_mean_shift.py \
#   --model_name google/gemma-3-12b-it \
#   --input_h5 ./process_generations/processed_generations_more_h5/new_gemma/4_trivia_gemma_extended_with_concat_3000_train/balanced/train_verbalised_embeddings.h5 \
#   --no-enable_brief \
#   --num_samples 30 \
#   --device cuda:0 \
#   --dtype bfloat16 \
#   --ablate_heads a47.h9,a47.h8,a27.h8,a27.h9,m46, \
#   --ablate_heads a47.h9,a47.h8,m46,a27.h8,a27.h9 \
#   --dataset trivia_qa \
#   --low_conf_threshold 0.2 \
#   --high_conf_threshold 0.8 \
#   --expected_guess_tokens 3 \
#   --expected_probability_tokens 5 \
#   --alpha 0.5 1.0 2.0 \
#   --intervention_site input \
#   --ablation_mode none probability_last_token_mean_replace extended_probability_last_token_mean_replace probability_pre_and_post_period_digit_mean_replace