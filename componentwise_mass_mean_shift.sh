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

# Mistral output experiments
# python3 ./componentwise_mass_mean_shift/run_componentwise_mass_mean_shift.py \
#   --model_name mistralai/Mistral-7B-Instruct-v0.1 \
#   --input_h5 ./process_generations/processed_generations_more_h5/3_math_mistral_extended_with_concat_train/train_verbalised_embeddings.h5 \
#   --direction_input_h5 ./process_generations/processed_generations_more_h5/mistral/2_200_9_prob_toks/train_verbalised_embeddings.h5 \
#   --direction_low_conf_threshold 0.1 \
#   --direction_high_conf_threshold 0.9 \
#   --no-enable_brief \
#   --num_samples 55 \
#   --device cuda:0 \
#   --dtype bfloat16 \
#   --ablate_heads a13,a14,m13,a11,m17 \
#   --dataset math \
#   --low_conf_threshold 0.0 \
#   --high_conf_threshold 1.0 \
#   --expected_guess_tokens 5 \
#   --expected_probability_tokens 7 \
#   --alpha -2.0 -1.0 -0.5 0.0 0.25 0.5 0.75 1.0 1.5 2.0 3.0 \
#   --intervention_site output \
#   --ablation_mode none last_a_panl_and_pc_mean_replace \

# python3 ./componentwise_mass_mean_shift/run_componentwise_mass_mean_shift.py \
#   --model_name google/gemma-3-12b-it \
#   --input_h5 ./process_generations/processed_generations_more_h5/1_math_gemma_extended_train/balanced/train_verbalised_embeddings.h5 \
#   --direction_input_h5 ./process_generations/processed_generations_more_h5/new_gemma/4_trivia_gemma_extended_with_concat_3000_train/balanced/train_verbalised_embeddings.h5 \
#   --direction_low_conf_threshold 0.2 \
#   --direction_high_conf_threshold 0.8 \
#   --no-enable_brief \
#   --num_samples 55 \
#   --device cuda:0 \
#   --dtype bfloat16 \
#   --ablate_heads m20,a21,a20,m29,a25,m25,a17,m21,m23,a22,m27,a14,m34,m24,m19 \
#   --dataset math \
#   --low_conf_threshold 0.1 \
#   --high_conf_threshold 0.9 \
#   --expected_guess_tokens 3 \
#   --expected_probability_tokens 5 \
#   --alpha -2.0 -1.0 -0.5 0.0 0.25 0.5 0.75 1.0 1.5 2.0 3.0 \
#   --intervention_site output \
#   --ablation_mode none last_a_panl_and_pc_mean_replace

# Mistral output experiments
# python3 ./componentwise_mass_mean_shift/run_componentwise_mass_mean_shift.py \
#   --model_name mistralai/Mistral-7B-Instruct-v0.1 \
#   --input_h5 ./ans_gen/generated_answers_with_processing/11_trivia_mistral_ling/train_verbalised_embeddings.h5 \
#   --direction_input_h5 ./process_generations/processed_generations_more_h5/mistral/2_200_9_prob_toks/train_verbalised_embeddings.h5 \
#   --direction_low_conf_threshold 0.1 \
#   --direction_high_conf_threshold 0.9 \
#   --no-enable_brief \
#   --num_samples 111 \
#   --device cuda:0 \
#   --dtype bfloat16 \
#   --ablate_heads a13,a14,m13,a11,m17,a16,m14,m16,m12,m11 \
#   --dataset trivia_qa \
#   --low_conf_threshold 0.1 \
#   --high_conf_threshold 0.9 \
#   --expected_guess_tokens 5 \
#   --expected_probability_tokens 7 \
#   --expected_confidence_tokens 5 \
#   --alpha 1.0 1.5 2.0 \
#   --intervention_site output \
#   --ablation_mode none generated_tokens_mean_replace \
#   --linguistic_confidence_prompt \
#   --semantic_similarity_method xverify
#   # --ablate_heads a3,a19,m5,a27,m8,a30,m15,m21,m24,m29 \

  # --direction_input_h5 ./process_generations/processed_generations_more_h5/mistral/2_200_9_prob_toks/train_verbalised_embeddings.h5 \
  # --direction_low_conf_threshold 0.1 \
  # --direction_high_conf_threshold 0.9 \
  # --input_h5 ./process_generations/processed_generations_more_h5/mistral/3_trivia_mistral_linguistic_train/balanced/train_verbalised_embeddings.h5 \
# python3 ./componentwise_mass_mean_shift/run_componentwise_mass_mean_shift.py \
#   --model_name mistralai/Mistral-7B-Instruct-v0.1 \
#   --input_h5 ./ans_gen/generated_answers_with_processing/1_trivia_mistral_correct/train_verbalised_embeddings.h5 \
#   --no-enable_brief \
#   --num_samples 111 \
#   --device cuda:0 \
#   --dtype bfloat16 \
#   --ablate_heads a13,a14,m13,a11,m17,a16,m14,m16,m12,m11,a12,a9,m9,a17,a8 \
#   --dataset trivia_qa \
#   --low_conf_threshold 0.1 \
#   --high_conf_threshold 0.9 \
#   --expected_guess_tokens 5 \
#   --expected_probability_tokens 7 \
#   --expected_confidence_tokens 5 \
#   --alpha 1.0 2.0 \
#   --intervention_site output \
#   --ablation_mode none generated_tokens_mean_replace \
#   --semantic_similarity_method xverify
#   --linguistic_confidence_prompt

# python3 ./componentwise_mass_mean_shift/run_componentwise_mass_mean_shift.py \
#   --model_name mistralai/Mistral-7B-Instruct-v0.1 \
#   --input_h5 ./ans_gen/generated_answers_with_processing/10_math_mistral/train_verbalised_embeddings.h5 \
#   --no-enable_brief \
#   --num_samples 111 \
#   --device cuda:0 \
#   --dtype bfloat16 \
#   --ablate_heads a13,a14,m13,a11,m17,a16,m14,m16,m12,m11,a12,a9,m9,a17,a8 \
#   --dataset math \
#   --low_conf_threshold 0.1 \
#   --high_conf_threshold 0.9 \
#   --expected_guess_tokens 5 \
#   --expected_probability_tokens 7 \
#   --expected_confidence_tokens 5 \
#   --alpha 1.0 2.0 \
#   --intervention_site output \
#   --ablation_mode none generated_tokens_mean_replace \
#   --semantic_similarity_method xverify

# python3 ./componentwise_mass_mean_shift/run_componentwise_mass_mean_shift.py \
#   --model_name mistralai/Mistral-7B-Instruct-v0.1 \
#   --input_h5 ./ans_gen/generated_answers_with_processing/10_math_mistral/train_verbalised_embeddings.h5 \
#   --direction_input_h5 ./ans_gen/generated_answers_with_processing/1_trivia_mistral_correct/train_verbalised_embeddings.h5 \
#   --direction_low_conf_threshold 0.1 \
#   --direction_high_conf_threshold 0.9 \
#   --no-enable_brief \
#   --num_samples 111 \
#   --device cuda:0 \
#   --dtype bfloat16 \
#   --ablate_heads a13,a14,m13,a11,m17 \
#   --dataset math \
#   --low_conf_threshold 0.0 \
#   --high_conf_threshold 1.0 \
#   --expected_guess_tokens 5 \
#   --expected_probability_tokens 7 \
#   --expected_confidence_tokens 5 \
#   --alpha 1.0 2.0 \
#   --intervention_site output \
#   --ablation_mode none generated_tokens_mean_replace \
#   --semantic_similarity_method xverify

python3 ./componentwise_mass_mean_shift/run_componentwise_mass_mean_shift.py \
  --model_name mistralai/Mistral-7B-Instruct-v0.1 \
  --input_h5 ./ans_gen/generated_answers_with_processing/10_math_mistral/train_verbalised_embeddings.h5 \
  --direction_input_h5 ./ans_gen/generated_answers_with_processing/1_trivia_mistral_correct/train_verbalised_embeddings.h5 \
  --direction_low_conf_threshold 0.1 \
  --direction_high_conf_threshold 0.9 \
  --no-enable_brief \
  --num_samples 111 \
  --device cuda:0 \
  --dtype bfloat16 \
  --ablate_heads a13,a14,m13,a11,m17,a16,m14,m16,m12,m11,a12,a9,m9,a17,a8 \
  --dataset math \
  --low_conf_threshold 0.0 \
  --high_conf_threshold 1.0 \
  --expected_guess_tokens 5 \
  --expected_probability_tokens 7 \
  --expected_confidence_tokens 5 \
  --alpha 1.0 2.0 \
  --intervention_site output \
  --ablation_mode none generated_tokens_mean_replace \
  --semantic_similarity_method xverify

# python3 ./componentwise_mass_mean_shift/run_componentwise_mass_mean_shift.py \
#   --model_name mistralai/Mistral-7B-Instruct-v0.1 \
#   --input_h5 ./ans_gen/generated_answers_with_processing/11_trivia_mistral_ling/train_verbalised_embeddings.h5 \
#   --direction_input_h5 ./ans_gen/generated_answers_with_processing/1_trivia_mistral_correct/train_verbalised_embeddings.h5 \
#   --direction_low_conf_threshold 0.1 \
#   --direction_high_conf_threshold 0.9 \
#   --no-enable_brief \
#   --num_samples 111 \
#   --device cuda:0 \
#   --dtype bfloat16 \
#   --ablate_heads a13,a14,m13,a11,m17,a16,m14,m16,m12,m11,a12,a9,m9,a17,a8 \
#   --dataset trivia_qa \
#   --low_conf_threshold 0.1 \
#   --high_conf_threshold 0.9 \
#   --expected_guess_tokens 5 \
#   --expected_probability_tokens 7 \
#   --expected_confidence_tokens 5 \
#   --alpha 1.0 2.0 \
#   --intervention_site output \
#   --ablation_mode none generated_tokens_mean_replace \
#   --semantic_similarity_method xverify \
#   --linguistic_confidence_prompt

# python3 ./componentwise_mass_mean_shift/run_componentwise_mass_mean_shift.py \
#   --model_name mistralai/Mistral-7B-Instruct-v0.1 \
#   --input_h5 ./ans_gen/generated_answers_with_processing/11_trivia_mistral_ling/train_verbalised_embeddings.h5 \
#   --direction_input_h5 ./ans_gen/generated_answers_with_processing/1_trivia_mistral_correct/train_verbalised_embeddings.h5 \
#   --direction_low_conf_threshold 0.1 \
#   --direction_high_conf_threshold 0.9 \
#   --no-enable_brief \
#   --num_samples 111 \
#   --device cuda:0 \
#   --dtype bfloat16 \
#   --ablate_heads a13,a14,m13,a11,m17 \
#   --dataset trivia_qa \
#   --low_conf_threshold 0.1 \
#   --high_conf_threshold 0.9 \
#   --expected_guess_tokens 5 \
#   --expected_probability_tokens 7 \
#   --expected_confidence_tokens 5 \
#   --alpha 1.0 2.0 \
#   --intervention_site output \
#   --ablation_mode none generated_tokens_mean_replace \
#   --semantic_similarity_method xverify \
#   --linguistic_confidence_prompt

# python3 ./componentwise_mass_mean_shift/run_componentwise_mass_mean_shift.py \
#   --model_name google/gemma-3-12b-it \
#   --input_h5 ./ans_gen/generated_answers_with_processing/7_trivia_gemma/train_verbalised_embeddings.h5 \
#   --no-enable_brief \
#   --num_samples 111 \
#   --device cuda:0 \
#   --dtype bfloat16 \
#   --ablate_heads m20,a21,a20,m29,a25,m25,a17,m21,m23,a22,m27,a14,m34,m24,m19 \
#   --dataset trivia_qa \
#   --low_conf_threshold 0.2 \
#   --high_conf_threshold 0.8 \
#   --expected_guess_tokens 3 \
#   --expected_probability_tokens 5 \
#   --alpha 1.0 2.0 \
#   --intervention_site output \
#   --ablation_mode none generated_tokens_mean_replace \
#   --semantic_similarity_method xverify

# python3 ./componentwise_mass_mean_shift/run_componentwise_mass_mean_shift.py \
#   --model_name google/gemma-3-12b-it \
#   --input_h5 ./ans_gen/generated_answers_with_processing/7_trivia_gemma/train_verbalised_embeddings.h5 \
#   --no-enable_brief \
#   --num_samples 111 \
#   --device cuda:0 \
#   --dtype bfloat16 \
#   --ablate_heads m20,a21,a20,m29,a25,m25,a17,m21,m23,a22 \
#   --dataset trivia_qa \
#   --low_conf_threshold 0.2 \
#   --high_conf_threshold 0.8 \
#   --expected_guess_tokens 3 \
#   --expected_probability_tokens 5 \
#   --alpha 1.0 2.0 \
#   --intervention_site output \
#   --ablation_mode none generated_tokens_mean_replace \
#   --semantic_similarity_method xverify

# python3 ./componentwise_mass_mean_shift/run_componentwise_mass_mean_shift.py \
#   --model_name google/gemma-3-12b-it \
#   --input_h5 ./ans_gen/generated_answers_with_processing/8_math_gemma/train_verbalised_embeddings.h5 \
#   --direction_input_h5 ./ans_gen/generated_answers_with_processing/7_trivia_gemma/train_verbalised_embeddings.h5 \
#   --direction_low_conf_threshold 0.2 \
#   --direction_high_conf_threshold 0.8 \
#   --no-enable_brief \
#   --num_samples 111 \
#   --device cuda:0 \
#   --dtype bfloat16 \
#   --ablate_heads m20,a21,a20,m29,a25,m25,a17,m21,m23,a22 \
#   --dataset math \
#   --low_conf_threshold 0.25 \
#   --high_conf_threshold 0.75 \
#   --expected_guess_tokens 3 \
#   --expected_probability_tokens 5 \
#   --alpha 1.0 2.0 \
#   --intervention_site output \
#   --ablation_mode none generated_tokens_mean_replace \
#   --semantic_similarity_method xverify

# python3 ./componentwise_mass_mean_shift/run_componentwise_mass_mean_shift.py \
#   --model_name google/gemma-3-12b-it \
#   --input_h5 ./ans_gen/generated_answers_with_processing/8_math_gemma/train_verbalised_embeddings.h5 \
#   --direction_input_h5 ./ans_gen/generated_answers_with_processing/7_trivia_gemma/train_verbalised_embeddings.h5 \
#   --direction_low_conf_threshold 0.2 \
#   --direction_high_conf_threshold 0.8 \
#   --no-enable_brief \
#   --num_samples 111 \
#   --device cuda:0 \
#   --dtype bfloat16 \
#   --ablate_heads m20,a21,a20,m29,a25,m25,a17,m21,m23,a22,m27,a14,m34,m24,m19 \
#   --dataset math \
#   --low_conf_threshold 0.25 \
#   --high_conf_threshold 0.75 \
#   --expected_guess_tokens 3 \
#   --expected_probability_tokens 5 \
#   --alpha 1.0 2.0 \
#   --intervention_site output \
#   --ablation_mode none generated_tokens_mean_replace \
#   --semantic_similarity_method xverify
