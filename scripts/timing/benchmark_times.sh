#!/bin/bash
#SBATCH -c 16
#SBATCH --mem 30g
#SBATCH --job-name benchT
#SBATCH --output=slrums/output_%A_%a.out # STDOUT
#SBATCH --gres=gpu:1
#SBATCH --array=0-11%3                    # 4 archs x 3 datasets; %4 = at most 4 GPUs of the node at once

# Controlled timing benchmark: unlearning of ONE class per (architecture, dataset)
# for every method reported in the paper (selection of src/plots/extract_methods.py,
# incl. NOISY = "Gaussian"), all on one node, each run timed individually.
#
# Isolation: runs from timing_benchmark/, where every top-level entry of the repo is
# symlinked EXCEPT the caches (mas/, fishers/, standard_cov_ledoit_wolf/) and models/.
#   - models_timing/<arch>_<ds> -> ../models/<arch>_<ds> (original checkpoint, read only)
#   - unlearned checkpoints go to timing_benchmark/models_timing/ and are deleted
#     after each run, so nothing under models/ is touched.
# Each method runs twice:
#   cold : its caches (MAS coefficients, distribution stats) deleted first, so the
#          one-off cost of computing them is included
#   warm : caches present, i.e. the marginal cost per extra class, as in the paper runs
# RETRAIN uses no cache and runs once.
#
# ONLY=<regex> runs only the methods whose label matches (e.g. ONLY=PARAMETRIC) and
# writes timing_benchmark/times/<arch>_<dataset>_<OUT_TAG>.csv (default OUT_TAG=subset),
# so the CSVs of a previous full run are not overwritten:
#   sbatch --export=ALL,ONLY=PARAMETRIC,OUT_TAG=parametric scripts/timing/benchmark_times.sh
# Output: timing_benchmark/times/<arch>_<dataset>.csv (one file per task: no
# concurrent appends to a shared CSV over NFS). Merge with
#   awk 'FNR>1||NR==1' timing_benchmark/times/*.csv > times_benchmark.csv

eval "$(conda shell.bash hook)"

conda activate unlearning

REPO=$(pwd)
export PYTHONPATH="$REPO"

archs=(resnet18 efficientnet_v2_s swin_t vit_b_16)
datasets=(cifar10 mnist cifar100)
experiment=${archs[$((SLURM_ARRAY_TASK_ID / 3))]}
dataset=${datasets[$((SLURM_ARRAY_TASK_ID % 3))]}

declare -A first_class=(["cifar10"]=0 ["mnist"]=0 ["cifar100"]=4)   # first class of the paper's lists
class=${CLASS:-${first_class[$dataset]}}

# SSD / EWC hyper-parameters selected per (arch, dataset) in selected_methods.csv
case "${experiment}_${dataset}" in
    efficientnet_v2_s_*)             ssd="0.1 1.0";  ewc="10.0" ;;
    resnet18_mnist)                  ssd="1.0 5.0";  ewc="10.0" ;;
    resnet18_*)                      ssd="1.0 10.0"; ewc="10.0" ;;
    swin_t_cifar10)                  ssd="0.1 1.0";  ewc="1.0"  ;;
    swin_t_*)                        ssd="0.1 1.0";  ewc="10.0" ;;
    vit_b_16_mnist)                  ssd="1.0 5.0";  ewc="1.0"  ;;
    vit_b_16_*)                      ssd="1.0 5.0";  ewc="10.0" ;;
esac
read -r ssd_l ssd_a <<< "$ssd"

SYN="--syntetic-samples 1000 --fixed-syntetic --emb-unfreeze embs --covariance-type ledoit_wolf"
# PARAMETRIC as selected for the paper: tiebreak by KS of ||x|| + DISTRIBUTIONAL test pipeline
# (same flags as unlearn_parametric.sh VARIANT=tiebreak_norm PIPELINE=tests)
PAR="--unlearning-method PARAMETRIC --cv-folds 5 --cv-repeats 3 --parametric-em-max-iter 20 --parametric-selection ll_tiebreak --parametric-tiebreak ks_norm --parametric-tie-margin 0.05 --parametric-pipeline tests --syntetic-samples 1000 --fixed-syntetic --emb-unfreeze embs --notop-margin 10"
# "label|flags"
methods=(
    "DISTRIBUTIONAL_notop|--unlearning-method DISTRIBUTIONAL --loss-forget notop ${SYN}"
    "DISTRIBUTIONAL_corrected_confidence_penalty|--unlearning-method DISTRIBUTIONAL --loss-forget corrected_confidence_penalty ${SYN}"
    "PARAMETRIC_notop|${PAR} --loss-forget notop"
    "PARAMETRIC_corrected_confidence_penalty|${PAR} --loss-forget corrected_confidence_penalty"
    "NOISY_notop|--unlearning-method NOISY --loss-forget notop ${SYN}"
    "NOISY_confidence_penalty|--unlearning-method NOISY --loss-forget confidence_penalty ${SYN}"
    "DELETE_embs|--unlearning-method DELETE --freeze-delete embs"
    "BOUNDARYSHRINK_eps0.1_lr1e-05_embs|--unlearning-method BOUNDARYSHRINK --freeze embs"
    "BOUNDARYEXPAND_lr1e-05_embs|--unlearning-method BOUNDARYEXPAND --freeze embs"
    "BADTEACHER_T1.0_forgetOnly_embs|--unlearning-method BADTEACHER --bt-temperature 1.0 --no-bt-use-retain --freeze embs"
    "RANDOMLABEL_noret_embs_forbidden|--unlearning-method RANDOMLABEL --rl-retain-fraction 0.0 --freeze embs --use-forbidden-labels"
    "SSD_MAS_lambda_${ssd_l}_alpha_${ssd_a}_embs|--unlearning-method SSD --SSD-coefficients MAS --SSD-lambda ${ssd_l} --SSD-alpha ${ssd_a} --freeze embs"
    "EWC_${ewc}_lambda_MAS_embsunfreeze_notop|--unlearning-method EWC --loss-forget notop --lambda-ewc ${ewc} --ewc-coefficients MAS --emb-unfreeze embs"
    "RETRAIN|--unlearning-method RETRAIN"
)

# ── isolated working directory ─────────────────────────────────────────
B="$REPO/timing_benchmark"
mkdir -p "$B/models_timing" "$B/times"
for entry in "$REPO"/*; do
    name=$(basename "$entry")
    case "$name" in
        timing_benchmark|models|mas|fishers|standard_cov_*|slrums) continue ;;
    esac
    [ -e "$B/$name" ] || ln -s "$entry" "$B/$name"
done
ln -sfn "$REPO/models/${experiment}_${dataset}" "$B/models_timing/${experiment}_${dataset}"
cd "$B"

ONLY=${ONLY:-}
OUT_TAG=${OUT_TAG:-${ONLY:+subset}}
out="times/${experiment}_${dataset}${OUT_TAG:+_${OUT_TAG}}.csv"
echo "node,gpu,architecture,dataset,class,method,cache,seconds,exit_code,slurm_task,start" > "$out"
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr "," " ")   # cgroup shows only our GPU

clear_cache() {   # caches are keyed by "<arch>_<dataset>_original_..." (cifar10_ != cifar100_)
    rm -f mas/${experiment}_${dataset}_original_* fishers/${experiment}_${dataset}_original_* \
          standard_cov_*/${experiment}_${dataset}_original_* 2>/dev/null
}

run_one() {   # $1 label  $2 cache mode  $3.. flags
    local label=$1 mode=$2; shift 2
    [ "$mode" == "cold" ] && clear_cache
    echo "Starting unlearning process for ${experiment} of ${dataset} with method $* [timing: ${mode}]"
    local t0 t1 rc start
    start=$(date -Iseconds); t0=$(date +%s.%N)
    python "$REPO/src/unlearning/main_unlearning.py" -o models_timing -e ${experiment}_${dataset} -m ${experiment} \
        --save-epoch 5 --max-epoch 30 -d datasets --dataset ${dataset} --data-mode CR \
        --train-batch-size 64 --val-batch-size 16 --seed 42 --classes-to-remove ${class} \
        --version original --epoch best "$@"
    rc=$?
    t1=$(date +%s.%N)
    printf "%s,%s,%s,%s,%s,%s,%s,%.1f,%d,%s,%s\n" "$(hostname)" "$gpu" "$experiment" "$dataset" "$class" \
        "$label" "$mode" "$(awk -v a="$t0" -v b="$t1" 'BEGIN{print b-a}')" "$rc" "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" "$start" >> "$out"
    rm -rf "models_timing/${experiment}_${dataset}_classes_to_remove_${class}"
}

for entry in "${methods[@]}"; do
    IFS='|' read -r label flags <<< "$entry"
    if [ -n "$ONLY" ] && ! [[ "$label" =~ $ONLY ]]; then continue; fi
    if [ "$label" == "RETRAIN" ]; then
        run_one "$label" none $flags
    else
        run_one "$label" cold $flags
        run_one "$label" warm $flags
    fi
done
echo "Done: ${experiment} ${dataset} class ${class} -> timing_benchmark/${out}"

conda deactivate
