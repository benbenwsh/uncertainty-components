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
#   python3 ./layerwise_mean_ablation/run_mean_ablation.py --input_h5 ./semantic_uncertainty/processed_generations_h5/3_train_200_samples_temp_0/train_verbalised_embeddings.h5 --num_samples 200 --device cuda:0 --ablate_layers "${layer}" --no-mean_from_low_confidence
# done
# python3 ./layerwise_mean_ablation/run_mean_ablation.py --input_h5 ./semantic_uncertainty/processed_generations_h5/3_train_200_samples_temp_0/train_verbalised_embeddings.h5 --num_samples 200 --device cuda:0 --ablate_layers 0-6 --no-mean_from_low_confidence

python3 ./blockwise_mean_ablation/run_blockwise_mean_ablation.py --input_h5 ./process_generations/processed_generations_more_h5/3_200_subblocks_one_more_prob_tok/train_verbalised_embeddings.h5 --no-enable_brief --num_samples 1000 --device cuda:0 --ablate_subblocks attn mlp --ablate_layers 10-16 --no-mean_from_low_confidence --ablation_mode none probability_tokens_mean_replace probability_last_token_mean_replace probability_span_except_last_token_mean_replace all_pre_probability_tokens_mean_replace guess_tokens_mean_replace all_pre_guess_tokens_mean_replace guess_then_guess_probability_zero_ablate