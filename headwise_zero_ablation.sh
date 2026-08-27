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

# Example with explicit unit selection (all listed heads/MLPs zeroed simultaneously
# on both high- and low-confidence groups):
# python3 ./headwise_zero_ablation/run_headwise_zero_ablation.py \
#   --input_h5 ./process_generations/processed_generations_more_h5/2_200_concat/train_verbalised_embeddings.h5 \
#   --no-enable_brief \
#   --num_samples 200 \
#   --device cuda:0 \
#   --ablate_heads a12.h3,a12.h7,a13.h2,m14 \
#   --ablation_mode none probability_tokens_zero_ablate

python3 ./headwise_zero_ablation/run_headwise_zero_ablation.py \
  --model_name google/gemma-3-12b-it \
  --input_h5 ./process_generations/processed_generations_more_h5/new_gemma/4_trivia_gemma_extended_with_concat_3000_train/balanced/train_verbalised_embeddings.h5 \
  --no-enable_brief \
  --num_samples 50 \
  --device cuda:0 \
  --dtype bfloat16 \
  --ablate_heads a47.h5,a43.h5,a45.h1,a43.h4,a44.h11,a35.h10,a32.h10,a34.h11,a39.h13,a35.h7,a35.h6,a46.h9,a39.h9,a46.h13,m38,a41.h9,m45,a47.h6,a45.h2,a47.h9 \
  --dataset trivia_qa \
  --expected_guess_tokens 3 \
  --expected_probability_tokens 5 \
  --low_conf_threshold 0.2 \
  --high_conf_threshold 0.8 \
  --ablation_mode none probability_pre_and_post_period_digit_zero_ablate extended_probability_last_token_zero_ablate probability_last_token_zero_ablate

