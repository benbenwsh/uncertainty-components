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

python3 ./ans_gen/generate_answers_h5.py \
  --model_name mistralai/Mistral-7B-Instruct-v0.1 \
  --num_samples 200 \
  --answerable_only \
  --collect_attn_block_embeddings \
  --collect_mlp_block_embeddings \
  --collect_qkvo_embeddings \
  --collect_concat_embeddings \
  --linguistic_confidence_prompt \
  --dataset trivia_qa
