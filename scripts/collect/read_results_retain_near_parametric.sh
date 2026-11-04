#!/bin/bash
#SBATCH -c 4
#SBATCH --job-name res_param_rn
#SBATCH --mem 10g
#SBATCH --gres=gpu:1
#SBATCH --output=slrums/output_%A_%a.out # STDOUT
#SBATCH --array=0%1

# Append the proposed PARAMETRIC runs (tiebreak_norm + tests) to the retain and
# NEAR (unknown) CSVs read by analyze_wilcoxon_unlearning_retain.py /
# analyze_wilcoxon_unlearning_unknown3.py.  --continue keeps the existing rows.
# NEAR exists only for single class (extract_NEAR_parametric.sh), so the
# multiple-class unknown analysis will still have no PARAMETRIC cells.
#   sbatch scripts/collect/read_results_retain_near_parametric.sh      (or: bash ... locally)

eval "$(conda shell.bash hook)"

conda activate unlearning
export PYTHONPATH='.'

TAG=PARAMETRIC_cv5x3_tbn_m0.05_tests

# retain
python src/collect/process_results_auto_retain.py --base-dir results --classes-filter single --include-version $TAG --results-file unlearning_results_single_embs_new_retain --online --continue
python src/collect/process_results_auto_retain.py --base-dir results --classes-filter multiple --include-version $TAG --results-file unlearning_results_multiple_embs_new_retain --online --continue

# NEAR / unknown (single only)
python src/collect/process_results_auto_NEAR.py --base-dir results --classes-filter single --include-version $TAG --results-file unlearning_results_single_embs_new_near --online --continue

for f in results/unlearning_results_{single,multiple}_embs_new_retain.csv results/unlearning_results_single_embs_new_near.csv; do
    echo "$(grep -c $TAG $f) $TAG rows in $f"
done

echo "Run ended"

conda deactivate
