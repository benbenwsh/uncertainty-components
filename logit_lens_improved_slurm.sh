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

python3 ./logit_lens_improved/run_logit_lens_improved.py \
  --input_h5 ./process_generations/processed_generations_more_h5/2/train_verbalised_embeddings.h5 \
  --n_examples 5 \
  --top_k 3 \
  --expected_guess_tokens 5 \
  --expected_probability_tokens 9 \
  --device cuda:0 \

# Example subblock run:
# python3 ./logit_lens_improved/run_logit_lens_improved.py \
#   --input_h5 ./process_generations/processed_generations_more_h5/2_200_concat/train_verbalised_embeddings.h5 \
#   --n_examples 20 \
#   --top_k 3 \
#   --expected_guess_tokens 5 \
#   --expected_probability_tokens 7 \
#   --subblock_mode \
#   --device cuda:0
