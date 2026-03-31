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

python3 ./verbalised_confidence_probes/train_semantic_probe_all_layers.py --train_path ./semantic_uncertainty/processed_generations/4/train_verbalised_embeddings.pkl --val_path ./semantic_uncertainty/processed_generations/5/validation_verbalised_embeddings.pkl --plot --n_negatives 99
