#!/bin/bash
#SBATCH -c 4
#SBATCH --job-name res_param
#SBATCH --mem 10g
#SBATCH --gres=gpu:1
#SBATCH --output=slrums/output_%A_%a.out # STDOUT
#SBATCH --array=0%1

# Process the extracted PARAMETRIC results (extract_parametric.sh) into their own
# CSV (not appended to unlearning_results_single_all_new.csv) and compare them
# with the matching DISTRIBUTIONAL runs.
#   sbatch --dependency=afterany:<JOBID_extract_parametric> scripts/collect/read_results_parametric.sh

eval "$(conda shell.bash hook)"

conda activate unlearning
export PYTHONPATH='.'

python src/collect/process_results_auto.py --base-dir results --classes-filter single --include-version PARAMETRIC_cv5 DISTRIBUTIONAL --results-file unlearning_results_single_parametric --online --continue

python src/collect/process_results_auto.py --base-dir results --classes-filter multiple --include-version PARAMETRIC_cv5 DISTRIBUTIONAL --results-file unlearning_results_multiple_parametric --online --continue

python src/analysis/compare_parametric.py --parametric results/unlearning_results_single_parametric.csv --reference unlearning_results_single_all_new.csv --epoch best

echo "Run ended"

conda deactivate
