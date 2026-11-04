#!/bin/bash
#SBATCH -c 4
#SBATCH --job-name abl_cmp
#SBATCH --mem 16g
#SBATCH --gres=gpu:1
#SBATCH --output=slrums/output_%j.out # STDOUT

# Paired ablation of the proposed method (src/analysis/ablation_compare.py):
#   sbatch --export=ALL,COMPARISON=noisy scripts/analysis/run_ablation_compare.sh    (or flujo)
# Output in results/ablation/<COMPARISON>/ (tables + plots/).

eval "$(conda shell.bash hook)"

conda activate unlearning
export PYTHONPATH='.'

python src/analysis/ablation_compare.py --comparison ${COMPARISON:-noisy}
echo "Run ended"

conda deactivate
