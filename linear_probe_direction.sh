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

# Linear / ridge verbalised-confidence probe directions on probability-span positions.
# --alpha values must be in [0, 1] (see linear_probe_direction/run_linear_probe_direction.py).
# Match --expected_probability_tokens to the number of tok_n_probability directories under --probe_dir.

python3 ./linear_probe_direction/run_linear_probe_direction.py \
  --input_h5 ./process_generations/processed_generations_more_h5/2_200_train_temp_0/train_verbalised_embeddings.h5 \
  --probe_dir ./verbalised_confidence_probes/results/mult_toks_all_layers/7_200 \
  --num_samples 200 \
  --device cuda:0 \
  --ablation_mode none probability_tokens_mean_replace \
  --ablate_layers 10-16 \
  --alpha -1 -0.5 0.0 0.25 0.5 0.75 1.0 1.5 2.0 \
  --ablation_targets low high \
  --no-enable_brief \
  --normalize_span_directions
