#!/bin/bash
#SBATCH -c 4
#SBATCH --job-name chosen_dist
#SBATCH --mem 16g
#SBATCH --output=slrums/output_%A_%a.out # STDOUT
#SBATCH --array=0%1

# Chosen distribution family per class (PARAMETRIC variants + DISTRIBUTIONAL)
# and norm / direction independence counts -> results/chosen_distributions.csv.
# Only reads the fit caches; no GPU needed.  Also runnable directly: bash scripts/analysis/export_chosen_distributions.sh

eval "$(conda shell.bash hook)"

conda activate unlearning
export PYTHONPATH='.'

python src/analysis/export_chosen_distributions.py --out results/chosen_distributions.csv | tee results/chosen_distributions.txt

echo "Run ended"

conda deactivate
