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
conda activate se_probes
source /vol/cuda/13.0.0/setup.sh
/usr/bin/nvidia-smi
uptime

# for layer in $(seq 0 31); do
#   python3 ./kv_patching_ablation/run_kv_patching_ablation.py --input_h5 ./process_generations/processed_generations_more_h5/1/train_verbalised_embeddings.h5 --num_samples 200 --device cuda:0 --ablate_layers "${layer}"
# done

python3 ./kv_patching_ablation/run_kv_patching_ablation.py \
  --input_h5 ./process_generations/processed_generations_more_h5/1_1000_qkvo_concat/train_verbalised_embeddings.h5 \
  --no-enable_brief \
  --num_samples 1000 \
  --device cuda:0 \
  --ablate_layers 10-31 \
  --ablation_mode none current_generated_token_after_prob_prefix_mean_replace probability_value_tokens_kv_mean_replace \
  --kv_patch_method last_query_span_read \
  --no-mean_from_low_confidence

# Simpler span-position K/V overwrite (approximation):
# python3 ./kv_patching_ablation/run_kv_patching_ablation.py \
#   --input_h5 ./process_generations/processed_generations_more_h5/1/train_verbalised_embeddings.h5 \
#   --no-enable_brief \
#   --num_samples 200 \
#   --device cuda:0 \
#   --ablate_layers 10-16 \
#   --ablation_mode none current_generated_token_after_prob_prefix_mean_replace \
#   --kv_patch_method span_key_positions
