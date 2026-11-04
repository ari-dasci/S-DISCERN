#!/bin/bash
#SBATCH -c 4
#SBATCH --job-name abl_flujo
#SBATCH --mem 16g
#SBATCH --gres=gpu:1
#SBATCH --output=slrums/output_%j.out # STDOUT

# Ablation of the PARAMETRIC selection flow (tests vs cv), see src/analysis/ablation_flujo.py.
# Output in results/ablation/flujo/.  Queued by hand after read_results_relaunch_embs.sh.

eval "$(conda shell.bash hook)"

conda activate unlearning
export PYTHONPATH='.'

python src/analysis/ablation_flujo.py
echo "Run ended"

conda deactivate
