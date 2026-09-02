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

# for layer in $(seq 0 31); do
#   python3 ./layerwise_mean_ablation/run_mean_ablation.py --input_h5 ./semantic_uncertainty/processed_generations_h5/3_train_200_samples_temp_0/train_verbalised_embeddings.h5 --num_samples 200 --device cuda:0 --ablate_layers "${layer}" --no-mean_from_low_confidence
# done
# python3 ./layerwise_mean_ablation/run_mean_ablation.py --input_h5 ./semantic_uncertainty/processed_generations_h5/3_train_200_samples_temp_0/train_verbalised_embeddings.h5 --num_samples 200 --device cuda:0 --ablate_layers 0-6 --no-mean_from_low_confidence

python3 ./mass_mean_probe/run_mass_mean_probe.py \
  --model_name google/gemma-3-12b-it \
  --input_h5 ./process_generations/processed_generations_more_h5/new_gemma/4_trivia_gemma_extended_with_concat_3000_train/balanced/train_verbalised_embeddings.h5 \
  --num_samples 50 \
  --device cuda:0 \
  --ablate_layers 0-47 \
  --dtype bfloat16 \
  --alpha 0.0 0.25 0.5 1.0 2.0 \
  --ablation_targets low high \
  --low_conf_threshold 0.2 \
  --high_conf_threshold 0.8 \
  --expected_guess_tokens 3 \
  --expected_probability_tokens 5 \
  --new_h5_format \
  --no-enable_brief \
  --ablation_mode none last_a_mean_replace last_a_and_panl_mean_replace last_a_panl_and_pc_mean_replace panl_mean_replace pc_mean_replace