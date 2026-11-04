#!/bin/bash
#SBATCH -c 16
#SBATCH --mem 20g
#SBATCH --job-name extract_missing
#SBATC#H -o ./slrums/%j.out # STDOUT
#SBATCH --output=slrums/output_%A_%a.out # STDOUT
#SBATCH --gres=gpu:1
#SBATCH --array=[0-1]%3

# Extracts embeddings/logits for "single class removal" runs whose model
# checkpoint exists under models/ but whose extraction is missing under
# results/. Built by diffing models/ vs results/ for the method families
# tracked in selected_methods.csv (see notes below for exclusions).
#
# NOT included here (verified against models/ + results/, not just the CSV):
#   - EWC_*_embsunfreeze for resnet18: CSV shows MISSING but results/.../best
#     already exists for lambda 1.0 and 10.0 -> selection-script bug, not a
#     missing extraction. Nothing to run.
#   - DISTRIBUTIONAL_..._forget_notop for swin_t: CSV shows MISSING and no
#     checkpoint exists under models/ for any class -> needs retraining first,
#     can't be recovered by extraction alone.

eval "$(conda shell.bash hook)"

conda activate unlearning
export PYTHONPATH='.'

methods=(
    "DISTRIBUTIONAL_1000_standard_cov_ledoit_wolf_syntetic_fixed_embs_forget_notop"
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

    # EWC grid: lambda in {1.0, 10.0}
    "EWC_1.0_lambda_MAS_embsunfreeze_notop"
    "EWC_10.0_lambda_MAS_embsunfreeze_notop"
)

methods=(
    "DISTRIBUTIONAL_1000_standard_cov_ledoit_wolf_syntetic_fixed_embs_forget_notop_margin_10.0"
    "DISTRIBUTIONAL_1000_standard_cov_ledoit_wolf_syntetic_fixed_embs_forget_notop"
)

experiments=("efficientnet_v2_s" "resnet18" "swin_t" "vit_b_16")

declare -A single_classes
single_classes["cifar10"]="0 1 2 3 4 5 6 7 8 9"
single_classes["mnist"]="0 1 2 3 4 5 6 7 8 9"
single_classes["cifar100"]="4 1 70 10 51 39 5 24 3 76 60 15 34 26 11 29 80 96 8 89" # Classes selected of cifar100

datasets=("cifar10" "mnist" "cifar100")

method=${methods[$SLURM_ARRAY_TASK_ID]}

for experiment in "${experiments[@]}"; do
    for dataset in "${datasets[@]}"; do
        classes=(${single_classes[$dataset]})
        for class in "${classes[@]}"; do
            model_dir="models/${experiment}_${dataset}_classes_to_remove_${class}"
            orig_best="${model_dir}/${method}/original_best"

            if [ ! -d "$orig_best" ]; then
                # Model was never trained (or checkpoint removed) for this combo, skip.
                continue
            fi

            checkpoints=()
            [ -f "${orig_best}/checkpoint_best.pt" ] && checkpoints+=("best")
            [ -f "${orig_best}/checkpoint_epoch_00.pt" ] && checkpoints+=("00")

            for checkpoint in "${checkpoints[@]}"; do
                results_csv="results/${experiment}_${dataset}_classes_to_remove_${class}/${method}/original_best/${checkpoint}/test.csv"
                if [ -f "$results_csv" ]; then
                    continue
                fi

                echo "Starting extraction process for ${experiment} on ${dataset} class ${class} with method ${method} checkpoint ${checkpoint}"
                python src/training/main_extractor.py -o results --model-directory models -e ${experiment}_${dataset} -m ${experiment} -d datasets --dataset ${dataset} --data-mode CR --classes-to-remove ${class} --unlearning-method ${method} --version original_best --checkpoint ${checkpoint}
            done
        done
    done
done

conda deactivate
