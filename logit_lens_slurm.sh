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

python3 ./logit_lens/h5_logit_lens_guess_all_layers.py --train_path ./semantic_uncertainty/processed_generations_h5/1_train_200_samples/train_verbalised_embeddings.h5 --val_path ./semantic_uncertainty/processed_generations_h5/2_val_200_samples/validation_verbalised_embeddings.h5 --plot --device cuda:0
