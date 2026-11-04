#!/bin/bash
#SBATCH -c 16
#SBATCH --mem 20g
#SBATCH --job-name extract_missing_multi
#SBATC#H -o ./slrums/%j.out # STDOUT
#SBATCH --output=slrums/output_%A_%a.out # STDOUT
#SBATCH --gres=gpu:1
#SBATCH --array=[0-20]%3

# Extracts embeddings/logits for "multiple class removal" runs (cifar100,
# 5/10/20/50 classes removed via seeds) whose model checkpoint exists under
# models/ but whose extraction is missing under results/. Classes-to-remove
# groups are discovered directly from the models/ directory names instead of
# being hardcoded (models/<exp>_cifar100_classes_to_remove_<c1>_<c2>_... ->
# --classes-to-remove <c1> <c2> ...), so it stays correct if new seed groups
# are added later.
#
# NOT included here (verified against models/ + results/, not just the CSV,
# and against the current parser/config code, not just what folders happen
# to exist):
#   - "EWC_*_embsunfreeze" and plain "DISTRIBUTIONAL_..._forget_notop" (no
#     margin) are OBSOLETE naming from an older version of the code:
#       * --emb-unfreeze now only accepts {freeze, embs, nofreeze} (see
#         src/config/parser.py) - "embsunfreeze" is not a legal value
#         anymore. Its modern equivalent is the "_embs_notop" folder name.
#       * --loss-forget notop now unconditionally appends
#         "_margin_{notop_margin}" (default 10.0) to the folder name (see
#         src/config/DISTRIBUTIONALconfig.py) - a plain "_forget_notop"
#         folder can no longer be produced. Its modern equivalent is
#         "_forget_notop_margin_10.0".
#     The methods list below already uses the modern names, so this script
#     picks up the real gap (already-trained checkpoints under the new
#     names that were never extracted) without trying to reproduce a naming
#     scheme the current code can't emit. See unlearn_missing_multi.sh for
#     the (small) genuine training gap this uncovered instead.
#
# Note: this also picks up efficientnet_v2_s and swin_t, which are fully
# trained for multi-class removal but have NO extraction at all yet, even
# though selected_methods.csv currently only tracks resnet18 and vit_b_16
# for the "multiple" scope.

eval "$(conda shell.bash hook)"

conda activate unlearning
export PYTHONPATH='.'

methods=(
    "DISTRIBUTIONAL_1000_standard_cov_ledoit_wolf_syntetic_fixed_embs_forget_notop_margin_10.0"
    "DISTRIBUTIONAL_1000_standard_cov_ledoit_wolf_syntetic_fixed_embs_forget_corrected_confidence_penalty"
    "DELETE_embs"
    "BOUNDARYSHRINK_eps0.1_lr1e-05_embs"
    "BOUNDARYEXPAND_lr1e-05_embs"
    "BADTEACHER_T1.0_forgetOnly_embs"
    "NOISY_1000_standard_cov_ledoit_wolf_syntetic_fixed_embs_forget_confidence_penalty"
    "NOISY_1000_standard_cov_ledoit_wolf_syntetic_fixed_embs_forget_notop"
    "RANDOMLABEL_noret_embs_forbidden"
    "RETRAIN"

    # SSD_MAS grid: lambda in {0.1, 0.5, 1.0} x alpha in {1.0, 5.0, 10.0}
    "SSD_MAS_lambda_0.1_alpha_1.0_embs"
    "SSD_MAS_lambda_0.1_alpha_5.0_embs"
    "SSD_MAS_lambda_0.1_alpha_10.0_embs"
    "SSD_MAS_lambda_0.5_alpha_1.0_embs"
    "SSD_MAS_lambda_0.5_alpha_5.0_embs"
    "SSD_MAS_lambda_0.5_alpha_10.0_embs"
    "SSD_MAS_lambda_1.0_alpha_1.0_embs"
    "SSD_MAS_lambda_1.0_alpha_5.0_embs"
    "SSD_MAS_lambda_1.0_alpha_10.0_embs"

    # EWC grid: lambda in {1.0, 10.0} (modern name: "_embs_", not "_embsunfreeze_")
    "EWC_1.0_lambda_MAS_embs_notop"
    "EWC_10.0_lambda_MAS_embs_notop"
)

experiments=("efficientnet_v2_s" "resnet18" "swin_t" "vit_b_16")
dataset="cifar100"

method=${methods[$SLURM_ARRAY_TASK_ID]}

for experiment in "${experiments[@]}"; do
    prefix="models/${experiment}_${dataset}_classes_to_remove_"
    for model_dir in ${prefix}*; do
        [ -d "$model_dir" ] || continue

        suffix="${model_dir#$prefix}"
        # Single-class dirs have no underscore in the suffix; skip those,
        # this script only handles multi-class (seed-based) removal.
        [[ "$suffix" == *_* ]] || continue

        classes_to_remove="${suffix//_/ }"
        orig_best="${model_dir}/${method}/original_best"

        if [ ! -d "$orig_best" ]; then
            continue
        fi

        checkpoints=()
        [ -f "${orig_best}/checkpoint_best.pt" ] && checkpoints+=("best")
        [ -f "${orig_best}/checkpoint_epoch_00.pt" ] && checkpoints+=("00")

        for checkpoint in "${checkpoints[@]}"; do
            results_csv="results/${experiment}_${dataset}_classes_to_remove_${suffix}/${method}/original_best/${checkpoint}/test.csv"
            if [ -f "$results_csv" ]; then
                continue
            fi

            echo "Starting extraction process for ${experiment} on ${dataset} classes [${classes_to_remove}] with method ${method} checkpoint ${checkpoint}"
            python src/training/main_extractor.py -o results --model-directory models -e ${experiment}_${dataset} -m ${experiment} -d datasets --dataset ${dataset} --data-mode CR --classes-to-remove ${classes_to_remove} --unlearning-method ${method} --version original_best --checkpoint ${checkpoint}
        done
    done
done

conda deactivate
