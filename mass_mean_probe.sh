#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=${USER} # required to send email notifcations - alternatively, enter an email address
#SBATCH --output=./slurm_out/slurm-%j.out
#SBATCH --mem=150G
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

  # --direction_input_h5 ./process_generations/processed_generations_more_h5/mistral/2_200_9_prob_toks/train_verbalised_embeddings.h5 \
  # --direction_low_conf_threshold 0.1 \
  # --direction_high_conf_threshold 0.9 \
# python3 ./mass_mean_probe/run_mass_mean_probe.py \
#   --model_name mistralai/Mistral-7B-Instruct-v0.1 \
#   --input_h5 ./ans_gen/generated_answers_with_processing/1_trivia_mistral_correct/train_verbalised_embeddings.h5 \
#   --num_samples 111 \
#   --device cuda:0 \
#   --ablate_layers 13,11,12,10,14 \
#   --dtype bfloat16 \
#   --alpha 1.0 2.0 \
#   --dataset trivia_qa \
#   --ablation_targets low high \
#   --low_conf_threshold 0.1 \
#   --high_conf_threshold 0.9 \
#   --expected_guess_tokens 5 \
#   --expected_probability_tokens 7 \
#   --new_h5_format \
#   --no-enable_brief \
#   --generated_tokens_source probability_prefix_first_token \
#   --ablation_mode none generated_tokens_mean_replace \
#   --semantic_similarity_method xverify

# python3 ./mass_mean_probe/run_mass_mean_probe.py \
#   --model_name mistralai/Mistral-7B-Instruct-v0.1 \
#   --input_h5 ./ans_gen/generated_answers_with_processing/10_math_mistral/train_verbalised_embeddings.h5 \
#   --direction_input_h5 ./ans_gen/generated_answers_with_processing/1_trivia_mistral_correct/train_verbalised_embeddings.h5 \
#   --direction_low_conf_threshold 0.1 \
#   --direction_high_conf_threshold 0.9 \
#   --num_samples 111 \
#   --device cuda:0 \
#   --ablate_layers 13,11,12,10,14 \
#   --dtype bfloat16 \
#   --alpha 1.0 2.0 \
#   --dataset math \
#   --ablation_targets low high \
#   --low_conf_threshold 0.0 \
#   --high_conf_threshold 1.0 \
#   --expected_guess_tokens 5 \
#   --expected_probability_tokens 7 \
#   --new_h5_format \
#   --no-enable_brief \
#   --generated_tokens_source probability_prefix_first_token \
#   --ablation_mode none generated_tokens_mean_replace \
#   --semantic_similarity_method xverify

# python3 ./mass_mean_probe/run_mass_mean_probe.py \
#   --model_name mistralai/Mistral-7B-Instruct-v0.1 \
#   --input_h5 ./ans_gen/generated_answers_with_processing/11_trivia_mistral_ling/train_verbalised_embeddings.h5 \
#   --direction_input_h5 ./ans_gen/generated_answers_with_processing/1_trivia_mistral_correct/train_verbalised_embeddings.h5 \
#   --direction_low_conf_threshold 0.1 \
#   --direction_high_conf_threshold 0.9 \
#   --num_samples 111 \
#   --device cuda:0 \
#   --ablate_layers 13,11,12,10,14 \
#   --dtype bfloat16 \
#   --alpha 1.0 2.0 \
#   --dataset trivia_qa \
#   --ablation_targets low high \
#   --low_conf_threshold 0.1 \
#   --high_conf_threshold 0.9 \
#   --expected_guess_tokens 5 \
#   --expected_probability_tokens 7 \
#   --expected_confidence_tokens 5 \
#   --new_h5_format \
#   --no-enable_brief \
#   --generated_tokens_source probability_prefix_first_token \
#   --ablation_mode none generated_tokens_mean_replace \
#   --semantic_similarity_method xverify \
#   --linguistic_confidence_prompt

# python3 ./mass_mean_probe/run_mass_mean_probe.py \
#   --model_name google/gemma-3-12b-it \
#   --input_h5 ./process_generations/processed_generations_more_h5/1_math_gemma_extended_train/balanced/train_verbalised_embeddings.h5 \
#   --direction_input_h5 ./process_generations/processed_generations_more_h5/new_gemma/4_trivia_gemma_extended_with_concat_3000_train/balanced/train_verbalised_embeddings.h5 \
#   --direction_low_conf_threshold 0.2 \
#   --direction_high_conf_threshold 0.8 \
#   --num_samples 55 \
#   --device cuda:0 \
#   --ablate_layers 21,24,22,23,20 \
#   --dtype bfloat16 \
#   --alpha 1.0 \
#   --dataset math \
#   --ablation_targets low high \
#   --low_conf_threshold 0.1 \
#   --high_conf_threshold 0.9 \
#   --expected_guess_tokens 3 \
#   --expected_probability_tokens 5 \
#   --new_h5_format \
#   --no-enable_brief \
#   --generated_tokens_source probability_prefix_first_token \
#   --ablation_mode none generated_tokens_mean_replace

# python3 ./mass_mean_probe/run_mass_mean_probe.py \
#   --model_name mistralai/Mistral-7B-Instruct-v0.1 \
#   --input_h5 ./ans_gen/generated_answers_with_processing/11_trivia_mistral_ling/train_verbalised_embeddings.h5 \
#   --direction_input_h5 ./process_generations/processed_generations_more_h5/mistral/2_200_9_prob_toks/train_verbalised_embeddings.h5 \
#   --direction_low_conf_threshold 0.1 \
#   --direction_high_conf_threshold 0.9 \
#   --num_samples 111 \
#   --device cuda:0 \
#   --ablate_layers 13,11,12 \
#   --dtype bfloat16 \
#   --alpha 1.0 2.0 \
#   --dataset trivia_qa \
#   --ablation_targets low high \
#   --low_conf_threshold 0.1 \
#   --high_conf_threshold 0.9 \
#   --expected_guess_tokens 5 \
#   --expected_probability_tokens 7 \
#   --expected_confidence_tokens 5 \
#   --new_h5_format \
#   --no-enable_brief \
#   --generated_tokens_source probability_prefix_first_token \
#   --ablation_mode none generated_tokens_mean_replace \
#   --linguistic_confidence_prompt \
#   --semantic_similarity_method xverify

# python3 ./mass_mean_probe/run_mass_mean_probe.py \
#   --model_name google/gemma-3-12b-it \
#   --input_h5 ./ans_gen/generated_answers_with_processing/8_math_gemma/train_verbalised_embeddings.h5 \
#   --direction_input_h5 ./ans_gen/generated_answers_with_processing/7_trivia_gemma/train_verbalised_embeddings.h5 \
#   --direction_low_conf_threshold 0.2 \
#   --direction_high_conf_threshold 0.8 \
#   --num_samples 111 \
#   --device cuda:0 \
#   --ablate_layers 21 \
#   --dtype bfloat16 \
#   --alpha 1.0 2.0 \
#   --dataset math \
#   --ablation_targets low high \
#   --low_conf_threshold 0.25 \
#   --high_conf_threshold 0.75 \
#   --expected_guess_tokens 3 \
#   --expected_probability_tokens 5 \
#   --new_h5_format \
#   --no-enable_brief \
#   --generated_tokens_source probability_prefix_first_token \
#   --ablation_mode none generated_tokens_mean_replace \
#   --semantic_similarity_method xverify