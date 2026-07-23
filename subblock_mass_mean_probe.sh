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

python3 ./subblock_mass_mean_probe/run_subblock_mass_mean_probe.py \
  --input_h5 ./process_generations/processed_generations_more_h5/1_200_with_prob_val/train_verbalised_embeddings.h5 \
  --num_samples 200 \
  --device cuda:0 \
  --ablate_layers 25-31 \
  --ablate_subblocks attn \
  --alpha 0.0 1.0 2.0 \
  --ablation_targets low high \
  --low_conf_threshold 0.1 \
  --high_conf_threshold 0.9 \
  --new_h5_format \
  --no-enable_brief \
  --ablation_mode none all_tokens_mean_replace generated_tokens_mean_replace  
