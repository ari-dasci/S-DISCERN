#!/bin/bash
#SBATCH -c 16
#SBATCH --mem 30g
#SBATCH --job-name unl_miss
#SBATCH --output=slrums/output_%A_%a.out # STDOUT
#SBATCH --gres=gpu:1
#SBATCH --array=[0-1]%2

# Trains the two "single class removal" combos that were verified (against
# models/ + results/, see extract_missing_single.sh) as never trained and
# never extracted at all, unlike every other combo in the same method family:
#
#   0: efficientnet_v2_s + NOISY_1000_standard_cov_ledoit_wolf_syntetic_fixed_embs_forget_confidence_penalty
#      (resnet18/swin_t/vit_b_16 already have this one; only efficientnet_v2_s is missing)
#   1: swin_t + DISTRIBUTIONAL_1000_standard_cov_ledoit_wolf_syntetic_fixed_embs_forget_notop
#      (efficientnet_v2_s/resnet18/vit_b_16 already have this one; only swin_t is missing)
#
# Command mirrors unlearn2.sh (same NOISY/DISTRIBUTIONAL flags, same
# hyperparameters: --save-epoch 5 --max-epoch 30 --seed 42 --version original
# --epoch best). Skips a class if its checkpoint already exists, so it's safe
# to re-submit if a task fails partway through.
#
# Deliberately excludes the multi-class (cifar100 5/10/20/50) gaps: those are
# already training via unlearn_multi_50.sh (job 169607) and queued for
# extraction via extract_multi.sh (job 169610, dependency=afterany:169607_*).

eval "$(conda shell.bash hook)"

conda activate unlearning
export PYTHONPATH='.'

experiments=(
    "efficientnet_v2_s"
    "swin_t"
)

unlearn_args=(
    "--unlearning-method NOISY --loss-forget confidence_penalty --syntetic-samples 1000 --fixed-syntetic --emb-unfreeze embs --covariance-type ledoit_wolf"
    "--unlearning-method DISTRIBUTIONAL --loss-forget notop --syntetic-samples 1000 --fixed-syntetic --emb-unfreeze embs --covariance-type ledoit_wolf"
)

method_names=(
    "NOISY_1000_standard_cov_ledoit_wolf_syntetic_fixed_embs_forget_confidence_penalty"
    "DISTRIBUTIONAL_1000_standard_cov_ledoit_wolf_syntetic_fixed_embs_forget_notop"
)

experiment=${experiments[$SLURM_ARRAY_TASK_ID]}
method=${unlearn_args[$SLURM_ARRAY_TASK_ID]}
method_name=${method_names[$SLURM_ARRAY_TASK_ID]}

datasets=("cifar10" "mnist")
classes=(0 1 2 3 4 5 6 7 8 9)
for dataset in "${datasets[@]}"; do
    for class in "${classes[@]}"; do
        ckpt="models/${experiment}_${dataset}_classes_to_remove_${class}/${method_name}/original_best/checkpoint_best.pt"
        if [ -f "$ckpt" ]; then
            echo "Skipping ${experiment} ${dataset} class ${class} (already trained)"
            continue
        fi
        echo "Starting unlearning process for ${experiment} of ${dataset} class ${class} with method ${method_name}"
        python src/unlearning/main_unlearning.py -o models -e ${experiment}_${dataset} -m ${experiment} --save-epoch 5 --max-epoch 30 -d datasets --dataset ${dataset} --data-mode CR --train-batch-size 64 --val-batch-size 16 --seed 42 --classes-to-remove ${class} --version original --epoch best ${method}
    done
done

datasets=("cifar100")
classes=(4 1 70 10 51 39 5 24 3 76 60 15 34 26 11 29 80 96 8 89) # Classes selected of cifar100
for dataset in "${datasets[@]}"; do
    for class in "${classes[@]}"; do
        ckpt="models/${experiment}_${dataset}_classes_to_remove_${class}/${method_name}/original_best/checkpoint_best.pt"
        if [ -f "$ckpt" ]; then
            echo "Skipping ${experiment} ${dataset} class ${class} (already trained)"
            continue
        fi
        echo "Starting unlearning process for ${experiment} of ${dataset} class ${class} with method ${method_name}"
        python src/unlearning/main_unlearning.py -o models -e ${experiment}_${dataset} -m ${experiment} --save-epoch 5 --max-epoch 30 -d datasets --dataset ${dataset} --data-mode CR --train-batch-size 64 --val-batch-size 16 --seed 42 --classes-to-remove ${class} --version original --epoch best ${method}
    done
done

conda deactivate
