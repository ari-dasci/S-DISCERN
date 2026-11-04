#!/bin/bash
#SBATCH -c 16
#SBATCH --mem 20g
#SBATCH --job-name hpx_param
#SBATCH --output=slrums/output_%A_%a.out # STDOUT
#SBATCH --gres=gpu:1
#SBATCH --array=0-51%4

# Extraction for the PARAMETRIC hyper-parameter mini study (checkpoints best / 00),
# same array layout as train_hparam_parametric.sh.  Skips what is already extracted.
#   sbatch --dependency=afterany:<train job> --export=ALL,VARIANT=...,PIPELINE=... scripts/hparam/extract_hparam_parametric.sh

eval "$(conda shell.bash hook)"

conda activate unlearning
export PYTHONPATH='.'

source scripts/hparam/hparam_parametric_common.sh
select_task "$SLURM_ARRAY_TASK_ID"
echo "### study=${study} method=${method} n_synth=${n_synth} margin=${margin} arch=${experiment}"

n_extracted=0; n_skipped=0; n_no_model=0
for dataset in "${DATASETS[@]}"; do
    for class in ${SINGLE_CLASSES[$dataset]}; do
        orig_best="models/${experiment}_${dataset}_classes_to_remove_${class}/${folder}/original_best"
        if [ ! -f "${orig_best}/checkpoint_best.pt" ]; then n_no_model=$((n_no_model + 1)); continue; fi
        checkpoints=("best")
        [ -f "${orig_best}/checkpoint_epoch_00.pt" ] && checkpoints+=("00")
        for checkpoint in "${checkpoints[@]}"; do
            if [ -f "results/${experiment}_${dataset}_classes_to_remove_${class}/${folder}/original_best/${checkpoint}/test.csv" ]; then
                n_skipped=$((n_skipped + 1)); continue
            fi
            echo "Extracting ${experiment} | ${dataset} | class ${class} | ${folder} | ckpt ${checkpoint}"
            n_extracted=$((n_extracted + 1))
            python src/training/main_extractor.py -o results --model-directory models \
                -e ${experiment}_${dataset} -m ${experiment} -d datasets --dataset ${dataset} \
                --data-mode CR --classes-to-remove ${class} --unlearning-method ${folder} \
                --version original_best --checkpoint ${checkpoint}
        done
    done
done
echo "Done: ${folder} ${experiment} (extracted: ${n_extracted}, already existed: ${n_skipped}, never trained: ${n_no_model})"

conda deactivate
