#!/bin/bash
#SBATCH -c 16
#SBATCH --mem 30g
#SBATCH --job-name trDNmiss
#SBATCH --output=slrums/output_%A_%a.out # STDOUT
#SBATCH --gres=gpu:1
#SBATCH --array=[0-4]%2

# Trains whatever is MISSING (if anything) for the DISTRIBUTIONAL/NOISY
# variants that src/plots/extract_methods.py selects as comparison references
# (selected_methods.csv, FIXED_METHODS list).
#
# Checked on 2026-09-23: every checkpoint below already exists under models/
# for all 4 architectures x 3 datasets x (10 or 20) classes. This script is
# therefore a no-op today -- it exists as a safety net for when a new
# architecture/dataset/class combo is added later, so you don't have to
# remember the exact training flags again. It skips (does not resubmit)
# anything whose checkpoint_best.pt already exists, so it's safe to re-run.
#
# Naming quirk: swin_t never converged with the plain "...forget_notop" run,
# so selected_methods.csv falls back to "...forget_notop_margin_10.0"
# (--notop-margin 10.0) for swin_t only; the other three architectures use the
# plain notop run. See resolve_notop_fallback() in src/plots/extract_methods.py.
#
# After any new checkpoint is trained here, run (in order):
#   1. extract_missing_dist_noisy_near.sh   (NEAR/"unknown" embeddings)
#   2. aggregate_near_after_ewc.sh          (rebuild the near CSV; generic
#      despite the name -- update its #SBATCH --dependency line, or drop it)
#   3. run_wilcoxon_all.sh

eval "$(conda shell.bash hook)"

conda activate unlearning
export PYTHONPATH='.'

ALL_ARCHS="resnet18 efficientnet_v2_s swin_t vit_b_16"
NOT_SWIN="resnet18 efficientnet_v2_s vit_b_16"

# folder_name|training_flags|applicable_architectures
# folder_name must match what main_unlearning.py derives from the flags, so
# the coverage check below looks in the right place under models/.
entries=(
    "DISTRIBUTIONAL_1000_standard_cov_ledoit_wolf_syntetic_fixed_embs_forget_notop|--unlearning-method DISTRIBUTIONAL --loss-forget notop --syntetic-samples 1000 --fixed-syntetic --emb-unfreeze embs --covariance-type ledoit_wolf|${NOT_SWIN}"
    "DISTRIBUTIONAL_1000_standard_cov_ledoit_wolf_syntetic_fixed_embs_forget_notop_margin_10.0|--unlearning-method DISTRIBUTIONAL --loss-forget notop --syntetic-samples 1000 --fixed-syntetic --emb-unfreeze embs --covariance-type ledoit_wolf --notop-margin 10.0|swin_t"
    "DISTRIBUTIONAL_1000_standard_cov_ledoit_wolf_syntetic_fixed_embs_forget_corrected_confidence_penalty|--unlearning-method DISTRIBUTIONAL --loss-forget corrected_confidence_penalty --syntetic-samples 1000 --fixed-syntetic --emb-unfreeze embs --covariance-type ledoit_wolf|${ALL_ARCHS}"
    "NOISY_1000_standard_cov_ledoit_wolf_syntetic_fixed_embs_forget_confidence_penalty|--unlearning-method NOISY --loss-forget confidence_penalty --syntetic-samples 1000 --fixed-syntetic --emb-unfreeze embs --covariance-type ledoit_wolf|${ALL_ARCHS}"
    "NOISY_1000_standard_cov_ledoit_wolf_syntetic_fixed_embs_forget_notop|--unlearning-method NOISY --loss-forget notop --syntetic-samples 1000 --fixed-syntetic --emb-unfreeze embs --covariance-type ledoit_wolf|${ALL_ARCHS}"
)

entry=${entries[$SLURM_ARRAY_TASK_ID]}
IFS='|' read -r folder_name method_flags archs_str <<< "$entry"
experiments=(${archs_str})
echo "### training: ${folder_name}   architectures: ${experiments[*]}"

datasets=("cifar10" "cifar100" "mnist")

declare -A single_classes
single_classes["cifar10"]="0 1 2 3 4 5 6 7 8 9"
single_classes["mnist"]="0 1 2 3 4 5 6 7 8 9"
single_classes["cifar100"]="4 1 70 10 51 39 5 24 3 76 60 15 34 26 11 29 80 96 8 89"

n_trained=0
n_skipped=0

for experiment in "${experiments[@]}"; do
    for dataset in "${datasets[@]}"; do
        classes=(${single_classes[$dataset]})

        for class in "${classes[@]}"; do
            ckpt="models/${experiment}_${dataset}_classes_to_remove_${class}/${folder_name}/original_best/checkpoint_best.pt"

            if [ -f "$ckpt" ]; then
                n_skipped=$((n_skipped + 1))
                continue
            fi

            echo "Training ${experiment} | ${dataset} | class ${class} | ${folder_name}"
            n_trained=$((n_trained + 1))
            python src/unlearning/main_unlearning.py -o models -e ${experiment}_${dataset} -m ${experiment} \
                --save-epoch 5 --max-epoch 30 -d datasets --dataset ${dataset} --data-mode CR \
                --train-batch-size 64 --val-batch-size 16 --seed 42 \
                --classes-to-remove ${class} --version original --epoch best \
                ${method_flags}
        done
    done
done

echo "Done: ${folder_name}  (trained: ${n_trained}, already existed: ${n_skipped})"

conda deactivate
