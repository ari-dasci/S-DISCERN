#!/bin/bash
#SBATCH -c 16
#SBATCH --mem 30g
#SBATCH --job-name rl_icu
#SBATCH --output=slrums/output_%A_%a.out # STDOUT
#SBATCH --gres=gpu:1
#SBATCH --array=0-11%4

# ICU relaunch of the "embs" runs (see relaunch_embs_common.sh): cifar100,
# 20 classes per seed, seeds 42-51 (the setting of the ICU figures, as in
# unlearn_ICU_20.sh).  main_unlearning_ICU.py writes
# results_ICU/<exp>/<version>/unlearning_results_ICU.csv itself and skips the
# steps already in it, so the job can be re-submitted.

eval "$(conda shell.bash hook)"

conda activate unlearning
export PYTHONPATH='.'

source scripts/unlearning/relaunch_embs_common.sh

dataset=cifar100
for seed in "${SEEDS[@]}"; do
    var="k20_seed_${seed}"
    classes=${!var}
    echo "ICU ${experiment} ${dataset} seed ${seed} with ${method_name}: ${classes}"
    python src/unlearning/main_unlearning_ICU.py -o models -e ${experiment}_${dataset} -m ${experiment} --dataset ${dataset} --seed ${seed} --classes-to-remove ${classes} ${TRAIN_COMMON} ${method_args}
done

echo "Run ended"

conda deactivate
