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

# Example: run one layer at a time with head selection per layer
# for layer in $(seq 10 16); do
#   python3 ./ind_head_kv_patching/run_ind_head_kv_patching.py \
#     --input_h5 ./process_generations/processed_generations_more_h5/1_1000_qkvo_concat/train_verbalised_embeddings.h5 \
#     --no-enable_brief \
#     --num_samples 1000 \
#     --device cuda:0 \
#     --ablate_heads_by_layer "${layer}.0,${layer}.1" \
#     --no-mean_from_low_confidence
# done

python3 ./ind_head_kv_patching/run_ind_head_kv_patching.py \
  --input_h5 ./process_generations/processed_generations_more_h5/2_200_concat/train_verbalised_embeddings.h5 \
  --no-enable_brief \
  --num_samples 200 \
  --device cuda:0 \
  --ablate_heads_by_layer 8.0,9.3,10.2,11.5,12.3,12.5,12.6,13.2,13.6,14.2,14.5,14.6,15.1,15.4,16.0,16.1,16.2,16.7 \
  # --ablate_heads_by_layer 8.2,9.12,10.10,11.21,12.13,12.22,12.24,12.25,12.26,13.11,13.12,13.24,13.26,14.8,14.11,14.21,14.23,14.26,15.7,15.19,16.0,16.7,16.11,16.31, \

# V-only variant:
# python3 ./ind_head_kv_patching/run_ind_head_kv_patching.py \
#   --input_h5 ./process_generations/processed_generations_more_h5/1_1000_qkvo_concat/train_verbalised_embeddings.h5 \
#   --no-enable_brief \
#   --num_samples 1000 \
#   --device cuda:0 \
#   --ablate_heads_by_layer 10.0,10.1,11.0,11.1,12.0,12.1,13.0,13.1,14.0,14.1,15.0,15.1,16.0,16.1 \
#   --ablate_v_only \
#   --no-mean_from_low_confidence
