#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=${USER}
#SBATCH --output=./slurm_out/slurm-%j.out

# CPU-only backfill of untitled, larger-font v2 run-folder PNGs from existing
# verbalised_confidence_probes/results/mult_toks_all_layers runs.
# Does not overwrite original PNGs; writes sibling *_v2.png files.

export PATH=/vol/bitbucket/${USER}/myvenv/bin/:$PATH
source ~/.bashrc
source activate
conda activate uncertainty_components
uptime

cd /vol/bitbucket/bhw22/uncertainty-components

python3 ./plotting/backfill_verbalised_multitoken_v2_plots.py
