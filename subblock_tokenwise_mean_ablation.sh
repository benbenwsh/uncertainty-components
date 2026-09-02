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

# Subblock tokenwise mean ablation (attn and/or mlp).
# --ablate_subblocks attn        -> ablate attn-out only
# --ablate_subblocks mlp         -> ablate mlp-out only
# --ablate_subblocks attn mlp    -> ablate both simultaneously

# Both attn+mlp simultaneously (individual-layer heatmap):
python3 ./subblock_tokenwise_mean_ablation/run_subblock_tokenwise_mean_ablation.py \
  --model_name google/gemma-3-12b-it \
  --input_h5 ./process_generations/processed_generations_more_h5/new_gemma/4_trivia_gemma_extended_with_concat_3000_train/balanced/train_verbalised_embeddings.h5 \
  --dataset trivia_qa \
  --no-enable_brief \
  --num_samples 20 \
  --device cuda:0 \
  --dtype bfloat16 \
  --new_h5_format \
  --ablate_subblocks mlp \
  --low_conf_threshold 0.2 \
  --high_conf_threshold 0.8 \
  --individual_layers \
  --expected_guess_tokens 3 \
  --expected_probability_tokens 5 \
  --extend_probability_span \
  --ablate_with_same_confidence \

# N.B. num_samples refer to the max number of iterations to perform, not limiting
# the mean vector computation
