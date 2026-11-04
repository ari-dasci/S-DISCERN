#!/bin/bash
#SBATCH -c 16
#SBATCH --mem 30g
#SBATCH --job-name rl_single
#SBATCH --output=slrums/output_%A_%a.out # STDOUT
#SBATCH --gres=gpu:1
#SBATCH --array=0-11%4

# Single-class relaunch of the "embs" runs (see relaunch_embs_common.sh):
# per class, train -> extract (best, 00) -> NEAR extraction (best, 00).
# One array task per (architecture, method); every step is skipped if its
# output already exists, so the job can be re-submitted.
# Queued by queue_relaunch_embs.sh (after relaunch_embs_prep.sh).

eval "$(conda shell.bash hook)"

conda activate unlearning
export PYTHONPATH='.'

source scripts/unlearning/relaunch_embs_common.sh

run_one () {
    local dataset=$1 class=$2
    local model_dir="models/${experiment}_${dataset}_classes_to_remove_${class}/${method_name}/original_best"
    if [ -f "${model_dir}/checkpoint_best.pt" ]; then
        echo "Skipping training ${experiment} ${dataset} class ${class} (already trained)"
    else
        echo "Unlearning ${experiment} ${dataset} class ${class} with ${method_name}"
        python src/unlearning/main_unlearning.py -o models -e ${experiment}_${dataset} -m ${experiment} --dataset ${dataset} --seed 42 --classes-to-remove ${class} ${TRAIN_COMMON} ${method_args}
    fi
    [ -d "$model_dir" ] || { echo "Missing ${model_dir} — not extracting"; return; }
    extract_ckpts results "$model_dir" ${experiment} ${dataset} ${class}
    extract_ckpts results_NEAR "$model_dir" ${experiment} ${dataset} ${class} --extract-dataset ${NEAR_DATASET[$dataset]}
}

for dataset in "cifar10" "mnist"; do
    for class in 0 1 2 3 4 5 6 7 8 9; do
        run_one ${dataset} ${class}
    done
done
for class in ${SINGLE_CIFAR100}; do
    run_one cifar100 ${class}
done

echo "Run ended"

conda deactivate
