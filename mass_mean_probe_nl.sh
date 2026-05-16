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

# Natural-language mass mean-direction probe (see mass_mean_probe_nl/run_mass_mean_probe_nl.py).
# Adjust --input_h5, --model_name, --ablate_layers, --ablation_mode, and --alpha as needed.

python3 ./mass_mean_probe_nl/run_mass_mean_probe_nl.py \
  --input_h5 ./process_generations/processed_generations_more_h5/3_200_subblocks_one_more_prob_tok/train_verbalised_embeddings.h5 \
  --num_samples 200 \
  --device cuda:0 \
  --ablate_layers 10-16 \
  --alpha -1 0.0 0.25 0.5 0.75 1.0 2.0 \
  --ablation_targets low high \
  --low_conf_threshold 0.1 \
  --high_conf_threshold 0.9 \
  --no-enable_brief \
  --new_h5_format
