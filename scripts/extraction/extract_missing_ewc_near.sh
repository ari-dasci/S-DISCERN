#!/bin/bash
#SBATCH -c 16
#SBATCH --mem 20g
#SBATCH --job-name exEWCnear
#SBATCH --output=slrums/output_%A_%a.out # STDOUT
#SBATCH --gres=gpu:1
#SBATCH --array=[0-3]%2

# Extracts the MISSING NEAR (OOD) embeddings for EWC.
#
# Why only NEAR: the plain extraction under results/ is already complete for
# every architecture (verified: 40 classes x 2 lambdas per architecture, all
# with test.csv), and both unlearning_results_single_embs_new.csv and
# ..._retain.csv already contain those rows. The only block missing EWC is the
# "unknown" one, which reads results_NEAR/ instead, and that directory has:
#
#   efficientnet_v2_s  EWC_*_embsunfreeze_notop   40/40  OK
#   swin_t             EWC_*_embsunfreeze_notop   40/40  OK
#   resnet18           EWC_*_embs_notop            0/40  <- missing
#   vit_b_16           EWC_*_embsunfreeze_notop    0/40  <- missing
#
# Note the naming split: resnet18 logged its unfrozen-embedding EWC runs as
# "MAS_embs_notop" while the other three used "MAS_embsunfreeze_notop". The
# original extract_NEAR.sh only listed the "embsunfreeze" spelling, which is
# why resnet18 was never extracted. This script loops over both spellings and
# skips whatever does not apply, so it is safe to re-run.
#
# After this job finishes, run: ./aggregate_near_after_ewc.sh

eval "$(conda shell.bash hook)"

conda activate unlearning
export PYTHONPATH='.'

methods=(
    "EWC_1.0_lambda_MAS_embs_notop"
    "EWC_10.0_lambda_MAS_embs_notop"
    "EWC_1.0_lambda_MAS_embsunfreeze_notop"
    "EWC_10.0_lambda_MAS_embsunfreeze_notop"
)

experiments=("resnet18" "vit_b_16" "efficientnet_v2_s" "swin_t")
datasets=("cifar10" "mnist" "cifar100")

# Several architectures have checkpoints under BOTH spellings, but only one of
# them is the variant that reaches the aggregated CSVs and therefore the one
# selected_methods.csv can pick. Extracting the other spelling is wasted GPU
# time, so by default each architecture only extracts its own.
# Set EXTRACT_ALL_SPELLINGS=1 to extract both everywhere instead.
EXTRACT_ALL_SPELLINGS=${EXTRACT_ALL_SPELLINGS:-0}

declare -A spelling
spelling["resnet18"]="embs_notop"
spelling["efficientnet_v2_s"]="embsunfreeze_notop"
spelling["swin_t"]="embsunfreeze_notop"
spelling["vit_b_16"]="embsunfreeze_notop"

declare -A single_classes
single_classes["cifar10"]="0 1 2 3 4 5 6 7 8 9"
single_classes["mnist"]="0 1 2 3 4 5 6 7 8 9"
single_classes["cifar100"]="4 1 70 10 51 39 5 24 3 76 60 15 34 26 11 29 80 96 8 89"

# Near-OOD counterpart used to extract the "unknown" embeddings.
declare -A near_dataset
near_dataset["cifar10"]="cifar100"
near_dataset["cifar100"]="cifar10"
near_dataset["mnist"]="letters"

method=${methods[$SLURM_ARRAY_TASK_ID]}
echo "### method: ${method}"

for experiment in "${experiments[@]}"; do
    if [ "$EXTRACT_ALL_SPELLINGS" != "1" ] && [[ "$method" != *"${spelling[$experiment]}" ]]; then
        echo "skip ${experiment}: uses ${spelling[$experiment]}, not this variant"
        continue
    fi

    for dataset in "${datasets[@]}"; do
        extract_dataset=${near_dataset[$dataset]}
        classes=(${single_classes[$dataset]})

        for class in "${classes[@]}"; do
            model_dir="models/${experiment}_${dataset}_classes_to_remove_${class}/${method}/original_best"

            if [ ! -d "$model_dir" ]; then
                # This architecture never ran this EWC spelling, nothing to do.
                continue
            fi

            checkpoints=()
            [ -f "${model_dir}/checkpoint_best.pt" ] && checkpoints+=("best")
            [ -f "${model_dir}/checkpoint_epoch_00.pt" ] && checkpoints+=("00")

            for checkpoint in "${checkpoints[@]}"; do
                near_csv="results_NEAR/${experiment}_${dataset}_classes_to_remove_${class}/${method}/original_best/${checkpoint}/test.csv"

                if [ -f "$near_csv" ]; then
                    continue
                fi

                echo "NEAR extraction: ${experiment} | ${dataset} (ood=${extract_dataset}) | class ${class} | ${method} | ckpt ${checkpoint}"
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

echo "Done: ${method}"

conda deactivate
