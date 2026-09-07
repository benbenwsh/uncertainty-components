#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=${USER}
#SBATCH --output=./slurm_out/slurm-%j.out

# CPU-only backfill of research-ready heatmap PNGs from existing results.
# Does not overwrite annotated heatmaps; writes sibling *_paper.png files.

export PATH=/vol/bitbucket/${USER}/myvenv/bin/:$PATH
source ~/.bashrc
source activate
conda activate uncertainty_components
uptime

cd /vol/bitbucket/bhw22/uncertainty-components

python3 ./plotting/backfill_tokenwise_paper_heatmaps.py
python3 ./plotting/backfill_dla_paper_heatmaps.py
