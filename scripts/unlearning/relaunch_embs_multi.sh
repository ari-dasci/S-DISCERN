#!/bin/bash
#SBATCH -c 16
#SBATCH --mem 30g
#SBATCH --job-name rl_multi
#SBATCH --output=slrums/output_%A_%a.out # STDOUT
#SBATCH --gres=gpu:1
#SBATCH --array=0-11%4

# Multi-class relaunch of the "embs" runs (see relaunch_embs_common.sh):
# cifar100, 5 / 10 / 20 / 50 classes at once, seeds 42-51 (same lists as
# unlearn_multi_{5,10,20,50}.sh); train -> extract (best, 00).
#   KS="5 10" restricts the numbers of classes (default "5 10 20 50").

eval "$(conda shell.bash hook)"

conda activate unlearning
export PYTHONPATH='.'

source scripts/unlearning/relaunch_embs_common.sh

dataset=cifar100
for k in ${KS:-5 10 20 50}; do
    for seed in "${SEEDS[@]}"; do
        var="k${k}_seed_${seed}"
        classes=${!var}
        model_dir="models/${experiment}_${dataset}_classes_to_remove_${classes// /_}/${method_name}/original_best"
        if [ -f "${model_dir}/checkpoint_best.pt" ]; then
            echo "Skipping training ${experiment} k=${k} seed ${seed} (already trained)"
        else
            echo "Unlearning ${experiment} ${dataset} k=${k} seed ${seed} with ${method_name}: ${classes}"
            python src/unlearning/main_unlearning.py -o models -e ${experiment}_${dataset} -m ${experiment} --dataset ${dataset} --seed ${seed} --classes-to-remove ${classes} ${TRAIN_COMMON} ${method_args}
        fi
        [ -d "$model_dir" ] || { echo "Missing ${model_dir} — not extracting"; continue; }
        extract_ckpts results "$model_dir" ${experiment} ${dataset} "${classes}"
    done
done

echo "Run ended"

conda deactivate
