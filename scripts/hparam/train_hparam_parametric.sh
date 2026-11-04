#!/bin/bash
#SBATCH -c 16
#SBATCH --mem 30g
#SBATCH --job-name hp_param
#SBATCH --output=slrums/output_%A_%a.out # STDOUT
#SBATCH --gres=gpu:1
#SBATCH --array=0-51%4

# PARAMETRIC hyper-parameter mini study, single class removal (see
# hparam_parametric_common.sh).  One array task per (config, architecture).
# Skips classes already trained.  Waits for the fit cache of the main runs.
#   sbatch --export=ALL,VARIANT=tiebreak_norm,PIPELINE=cv scripts/hparam/train_hparam_parametric.sh

eval "$(conda shell.bash hook)"

conda activate unlearning
export PYTHONPATH='.'

source scripts/hparam/hparam_parametric_common.sh
select_task "$SLURM_ARRAY_TASK_ID"
echo "### study=${study} method=${method} n_synth=${n_synth} margin=${margin} arch=${experiment}"
echo "### folder: ${folder}"

n_trained=0; n_skipped=0
for dataset in "${DATASETS[@]}"; do
    wait_cache "$experiment" "$dataset" || continue
    for class in ${SINGLE_CLASSES[$dataset]}; do
        if [ -f "models/${experiment}_${dataset}_classes_to_remove_${class}/${folder}/original_best/checkpoint_best.pt" ]; then
            n_skipped=$((n_skipped + 1)); continue
        fi
        echo "Training ${experiment} | ${dataset} | class ${class} | ${folder}"
        n_trained=$((n_trained + 1))
        python src/unlearning/main_unlearning.py -o models -e ${experiment}_${dataset} -m ${experiment} \
            --save-epoch 5 --max-epoch 30 -d datasets --dataset ${dataset} --data-mode CR \
            --train-batch-size 64 --val-batch-size 16 --seed 42 \
            --classes-to-remove ${class} --version original --epoch best \
            ${flags}
    done
done
echo "Done: ${folder} ${experiment} (trained: ${n_trained}, already existed: ${n_skipped})"

conda deactivate
