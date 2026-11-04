#!/bin/bash
#SBATCH -c 4
#SBATCH --job-name rl_read
#SBATCH --mem 10g
#SBATCH --gres=gpu:1
#SBATCH --output=slrums/output_%A_%a.out # STDOUT
#SBATCH --array=0%1

# Append the relaunched "embs" runs (relaunch_embs_*.sh) to the processed CSVs
# used by the analyses (same files as read_results.sh), and rebuild the ICU
# aggregate.  --continue skips rows already present; the old runs were renamed
# (and their CSV rows rewritten) by rename_embs_mislabeled.sh, so the new
# "_embs" rows do not collide with them.

eval "$(conda shell.bash hook)"

conda activate unlearning
export PYTHONPATH='.'

VERSIONS=$(bash -c 'SLURM_ARRAY_TASK_ID=0; source scripts/unlearning/relaunch_embs_common.sh >/dev/null; for m in "${METHODS[@]}"; do echo "${m%%|*}"; done')

for filt in single multiple; do
    python src/collect/process_results_auto.py --base-dir results --classes-filter ${filt} --include-version ${VERSIONS} --results-file unlearning_results_${filt}_all_new --online --continue
    python src/collect/process_results_auto.py --base-dir results --classes-filter ${filt} --include-version ${VERSIONS} --results-file unlearning_results_${filt}_embs_new --online --continue
    python src/collect/process_results_auto_retain.py --base-dir results --classes-filter ${filt} --include-version ${VERSIONS} --results-file unlearning_results_${filt}_embs_new_retain --online --continue
done
python src/collect/process_results_auto_NEAR.py --base-dir results --classes-filter single --include-version ${VERSIONS} --results-file unlearning_results_single_embs_new_near --online --continue

python src/analysis/analyze_ICU.py results_ICU

echo "Run ended"

conda deactivate
