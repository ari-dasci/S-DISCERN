#!/bin/bash
#SBATCH -c 4
#SBATCH --mem 8g
#SBATCH --job-name noisyVsDist
#SBATCH --output=slrums/output_%A.out # STDOUT

# Paired significance tests NOISY vs DISTRIBUTIONAL (notop and confidence_penalty
# pairs) for the three scenarios (single, multiple, ICU), each analysed on its own.
# See the docstring of src/analysis/analyze_noisy_vs_distributional.py for the design.
#
# CPU only (pandas/scipy), no GPU needed: it can also be run directly with
# `bash scripts/analysis/run_noisy_vs_distributional.sh`.
#
# Outputs in results/wilcox_noisy_vs_dist/ (a fresh directory, nothing existing
# is overwritten): tests_<scenario>.csv, summary_<scenario>.csv,
# tests_all_scenarios.csv, pairs_long.csv, coverage.csv

eval "$(conda shell.bash hook)"

conda activate unlearning
export PYTHONPATH='.'

python3 src/analysis/analyze_noisy_vs_distributional.py \
    --scenarios single multiple icu \
    --analysis-set both \
    --icu-unit both \
    --icu-run-aggregation auc \
    --alpha 0.05 \
    --min-pairs 6 \
    --output-dir results/wilcox_noisy_vs_dist

echo "Run ended"

conda deactivate
