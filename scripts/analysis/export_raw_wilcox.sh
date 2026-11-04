#!/bin/bash
# Export the raw per-class increments of the latent-space Wilcoxon analysis
# (single-class forgetting, no row dropped, is_outlier only flagged) to
# results/wilcox/raw/, then run the sanity checks. No test output is written:
# each analyze script exits right after building its delta tables.
#
#   bash scripts/analysis/export_raw_wilcox.sh

# no "set -u": conda's activate script reads unset vars (PS1) and fails under it

eval "$(conda shell.bash hook)"
conda activate unlearning
export PYTHONPATH='.'

OUT=results/wilcox/raw

# Same filters as the reference pass of run_wilcoxon_all.sh (--goodones,
# selected_methods.csv, DISTRIBUTIONAL/NOISY kept so they can be exported).
for script in src/analysis/analyze_wilcoxon_unlearning.py \
              src/analysis/analyze_wilcoxon_unlearning_retain.py \
              src/analysis/analyze_wilcoxon_unlearning_unknown3.py; do
    echo "--- $script"
    python3 "$script" --goodones --export-raw-dir "$OUT" 2>&1 \
        | grep -vE "warnings.warn|UserWarning|ConvergenceWarning"
done

echo
python3 src/analysis/check_raw_values.py

conda deactivate
