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

python3 ./componentwise_input_mass_mean_shift/run_componentwise_input_mass_mean_shift.py \
  --model_name google/gemma-3-12b-it \
  --input_h5 ./process_generations/processed_generations_more_h5/new_gemma/4_trivia_gemma_extended_with_concat_3000_train/balanced/train_verbalised_embeddings.h5 \
  --no-enable_brief \
  --num_samples 30 \
  --device cuda:0 \
  --dtype bfloat16 \
  --ablate_heads m42,m47,a45.h3,m44,m43,a47.h8,a47.h7,m33,m40,a43.h9
  --dataset trivia_qa \
  --low_conf_threshold 0.2 \
  --high_conf_threshold 0.8 \
  --expected_guess_tokens 3 \
  --expected_probability_tokens 5 \
  --alpha 0.5 1.0 1.5 \
  --ablation_mode none probability_pre_and_post_period_digit_mean_replace extended_probability_last_token_mean_replace probability_last_token_mean_replace
