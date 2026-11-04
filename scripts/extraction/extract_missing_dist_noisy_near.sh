#!/bin/bash
#SBATCH -c 16
#SBATCH --mem 20g
#SBATCH --job-name exDNnear
#SBATCH --output=slrums/output_%A_%a.out # STDOUT
#SBATCH --gres=gpu:1
#SBATCH --array=[0-4]%2

# Extracts the MISSING NEAR (OOD / "unknown" block) embeddings for the
# DISTRIBUTIONAL and NOISY variants that src/plots/extract_methods.py selects as
# comparison references (selected_methods.csv, FIXED_METHODS list).
#
# Verified before writing this (2026-09-23): every checkpoint below already
# exists under models/, and the plain "forget"/"retain" extraction under
# results/ is already complete for all 4 architectures (run
# train_missing_dist_noisy.sh first only if that check has changed since --
# it is a no-op right now). Only results_NEAR/ is missing, which is what this
# script fills in.
#
# Naming quirk this script accounts for: swin_t never converged with the
# plain "...forget_notop" DISTRIBUTIONAL run, so selected_methods.csv
# resolves swin_t to "...forget_notop_margin_10.0" instead (see
# resolve_notop_fallback() in src/plots/extract_methods.py); the other three
# architectures use the plain name. Each entry below is tagged with the
# architectures it actually applies to, so the array only extracts what
# belongs to each one -- no wasted GPU time on the wrong architecture.
#
# After this finishes:
#   1. Rerun the NEAR aggregation: aggregate_near_after_ewc.sh works
#      unchanged (nothing in it is EWC-specific, it just rescans
#      results/ + results_NEAR/) -- update its #SBATCH --dependency line to
#      this job's id, or drop the line and run it manually.
#   2. run_wilcoxon_all.sh

eval "$(conda shell.bash hook)"

conda activate unlearning
export PYTHONPATH='.'

ALL_ARCHS="resnet18 efficientnet_v2_s swin_t vit_b_16"
NOT_SWIN="resnet18 efficientnet_v2_s vit_b_16"

# method_string|applicable_architectures
entries=(
    "DISTRIBUTIONAL_1000_standard_cov_ledoit_wolf_syntetic_fixed_embs_forget_notop|${NOT_SWIN}"
    "DISTRIBUTIONAL_1000_standard_cov_ledoit_wolf_syntetic_fixed_embs_forget_notop_margin_10.0|swin_t"
    "DISTRIBUTIONAL_1000_standard_cov_ledoit_wolf_syntetic_fixed_embs_forget_corrected_confidence_penalty|${ALL_ARCHS}"
    "NOISY_1000_standard_cov_ledoit_wolf_syntetic_fixed_embs_forget_confidence_penalty|${ALL_ARCHS}"
    "NOISY_1000_standard_cov_ledoit_wolf_syntetic_fixed_embs_forget_notop|${ALL_ARCHS}"
)

entry=${entries[$SLURM_ARRAY_TASK_ID]}
method="${entry%%|*}"
experiments=(${entry#*|})
echo "### method: ${method}   architectures: ${experiments[*]}"

datasets=("cifar10" "cifar100" "mnist")

declare -A single_classes
single_classes["cifar10"]="0 1 2 3 4 5 6 7 8 9"
single_classes["mnist"]="0 1 2 3 4 5 6 7 8 9"
single_classes["cifar100"]="4 1 70 10 51 39 5 24 3 76 60 15 34 26 11 29 80 96 8 89"

# Near-OOD counterpart used to extract the "unknown" embeddings, same mapping
# as extract_NEAR.sh / extract_missing_ewc_near.sh.
declare -A near_dataset
near_dataset["cifar10"]="cifar100"
near_dataset["cifar100"]="cifar10"
near_dataset["mnist"]="letters"

n_extracted=0
n_skipped=0
n_no_checkpoint=0

for experiment in "${experiments[@]}"; do
    for dataset in "${datasets[@]}"; do
        extract_dataset=${near_dataset[$dataset]}
        classes=(${single_classes[$dataset]})

        for class in "${classes[@]}"; do
            model_dir="models/${experiment}_${dataset}_classes_to_remove_${class}/${method}/original_best"

            if [ ! -d "$model_dir" ]; then
                n_no_checkpoint=$((n_no_checkpoint + 1))
                continue
            fi

            checkpoints=()
            [ -f "${model_dir}/checkpoint_best.pt" ] && checkpoints+=("best")
            [ -f "${model_dir}/checkpoint_epoch_00.pt" ] && checkpoints+=("00")

            for checkpoint in "${checkpoints[@]}"; do
                near_csv="results_NEAR/${experiment}_${dataset}_classes_to_remove_${class}/${method}/original_best/${checkpoint}/test.csv"

                if [ -f "$near_csv" ]; then
                    n_skipped=$((n_skipped + 1))
                    continue
                fi

                echo "NEAR extraction: ${experiment} | ${dataset} (ood=${extract_dataset}) | class ${class} | ${method} | ckpt ${checkpoint}"
                n_extracted=$((n_extracted + 1))
                python src/training/main_extractor.py \
                    -o results_NEAR \
                    --model-directory models \
                    -e ${experiment}_${dataset} \
                    -m ${experiment} \
                    -d datasets \
                    --dataset ${dataset} \
                    --extract-dataset ${extract_dataset} \
                    --data-mode CR \
                    --classes-to-remove ${class} \
                    --unlearning-method ${method} \
                    --version original_best \
                    --checkpoint ${checkpoint}
            done
        done
    done
done

echo "Done: ${method}  (extracted: ${n_extracted}, already existed: ${n_skipped}, no checkpoint: ${n_no_checkpoint})"

conda deactivate
