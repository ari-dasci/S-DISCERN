# Sourced by relaunch_embs_{single,multi,ICU}.sh.
#
# Relaunch of the "embs" runs (classifier + LAST backbone stage) that were not
# really "embs" before patches/embs_last_stage.patch (see rename_embs_mislabeled.sh):
#   efficientnet_v2_s : every method of selected_methods.csv that uses "embs"
#   swin_t            : only SSD (its old "embs" filter only touched the classifier)
# EWC / SSD: the current winners of selected_methods.csv
# (EWC_10.0_lambda_MAS_embs_notop, SSD_MAS_lambda_0.1_alpha_1.0_embs).
# RETRAIN / original do not depend on the freezing and are not relaunched.

# "<version folder>|<arguments>"
METHODS=(
    "BADTEACHER_T1.0_forgetOnly_embs|--unlearning-method BADTEACHER --bt-temperature 1.0 --no-bt-use-retain --freeze embs"
    "BOUNDARYEXPAND_lr1e-05_embs|--unlearning-method BOUNDARYEXPAND --freeze embs"
    "BOUNDARYSHRINK_eps0.1_lr1e-05_embs|--unlearning-method BOUNDARYSHRINK --freeze embs"
    "DELETE_embs|--unlearning-method DELETE --freeze-delete embs"
    "RANDOMLABEL_noret_embs_forbidden|--unlearning-method RANDOMLABEL --rl-retain-fraction 0.0 --freeze embs --use-forbidden-labels"
    "NOISY_1000_standard_cov_ledoit_wolf_syntetic_fixed_embs_forget_notop|--unlearning-method NOISY --loss-forget notop --syntetic-samples 1000 --fixed-syntetic --emb-unfreeze embs --covariance-type ledoit_wolf"
    "NOISY_1000_standard_cov_ledoit_wolf_syntetic_fixed_embs_forget_confidence_penalty|--unlearning-method NOISY --loss-forget confidence_penalty --syntetic-samples 1000 --fixed-syntetic --emb-unfreeze embs --covariance-type ledoit_wolf"
    "PARAMETRIC_cv5x3_tbn_m0.05_tests_1000_standard_syntetic_fixed_embs_forget_notop_margin_10.0|--unlearning-method PARAMETRIC --cv-folds 5 --cv-repeats 3 --parametric-em-max-iter 20 --parametric-selection ll_tiebreak --parametric-tiebreak ks_norm --parametric-tie-margin 0.05 --parametric-pipeline tests --syntetic-samples 1000 --fixed-syntetic --emb-unfreeze embs --notop-margin 10 --loss-forget notop"
    "PARAMETRIC_cv5x3_tbn_m0.05_tests_1000_standard_syntetic_fixed_embs_forget_corrected_confidence_penalty|--unlearning-method PARAMETRIC --cv-folds 5 --cv-repeats 3 --parametric-em-max-iter 20 --parametric-selection ll_tiebreak --parametric-tiebreak ks_norm --parametric-tie-margin 0.05 --parametric-pipeline tests --syntetic-samples 1000 --fixed-syntetic --emb-unfreeze embs --notop-margin 10 --loss-forget corrected_confidence_penalty"
    "EWC_10.0_lambda_MAS_embs_notop|--unlearning-method EWC --loss-forget notop --lambda-ewc 10.0 --ewc-coefficients MAS --emb-unfreeze embs"
    "SSD_MAS_lambda_0.1_alpha_1.0_embs|--unlearning-method SSD --SSD-coefficients MAS --SSD-lambda 0.1 --SSD-alpha 1.0 --freeze embs"
    # Ablation of the selection flow (src/analysis/ablation_flujo.py): same PARAMETRIC
    # without the test cascade (CV duel for every class).  Appended at the end so the
    # indices above (already queued) do not change.
    "PARAMETRIC_cv5x3_tbn_m0.05_1000_standard_syntetic_fixed_embs_forget_notop_margin_10.0|--unlearning-method PARAMETRIC --cv-folds 5 --cv-repeats 3 --parametric-em-max-iter 20 --parametric-selection ll_tiebreak --parametric-tiebreak ks_norm --parametric-tie-margin 0.05 --syntetic-samples 1000 --fixed-syntetic --emb-unfreeze embs --notop-margin 10 --loss-forget notop"
    "PARAMETRIC_cv5x3_tbn_m0.05_1000_standard_syntetic_fixed_embs_forget_corrected_confidence_penalty|--unlearning-method PARAMETRIC --cv-folds 5 --cv-repeats 3 --parametric-em-max-iter 20 --parametric-selection ll_tiebreak --parametric-tiebreak ks_norm --parametric-tie-margin 0.05 --syntetic-samples 1000 --fixed-syntetic --emb-unfreeze embs --notop-margin 10 --loss-forget corrected_confidence_penalty"
)

# One array task per "<architecture> <index in METHODS>"  (array 0-11)
TASKS=(
    "efficientnet_v2_s 0" "efficientnet_v2_s 1" "efficientnet_v2_s 2" "efficientnet_v2_s 3"
    "efficientnet_v2_s 4" "efficientnet_v2_s 5" "efficientnet_v2_s 6" "efficientnet_v2_s 7"
    "efficientnet_v2_s 8" "efficientnet_v2_s 9" "efficientnet_v2_s 10"
    "swin_t 10"
    "efficientnet_v2_s 11" "efficientnet_v2_s 12"    # ablation_flujo: PARAMETRIC cv flow (array 12-13)
)

read -r experiment midx <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"
method_name=${METHODS[$midx]%%|*}
method_args=${METHODS[$midx]#*|}
echo "Task ${SLURM_ARRAY_TASK_ID}: ${experiment} ${method_name}"

# The fixed emb_freeze() / SSD filter must be in place (relaunch_embs_prep.sh)
if ! grep -q "_EMB_TRAINABLE_PREFIXES" src/models/ClassificationModel.py; then
    echo "patches/embs_last_stage.patch is not applied — aborting"; exit 1
fi

# Single class: cifar10 / mnist all classes, cifar100 the 20 selected classes
SINGLE_CIFAR100="4 1 70 10 51 39 5 24 3 76 60 15 34 26 11 29 80 96 8 89"
declare -A NEAR_DATASET=(["cifar10"]="cifar100" ["cifar100"]="cifar10" ["mnist"]="letters")

SEEDS=(42 43 44 45 46 47 48 49 50 51)
# Multi-class / ICU class lists (from unlearn_multi_{5,10,20,50}.sh / unlearn_ICU_20.sh)
k5_seed_42="81 14 3 94 35"
k5_seed_43="4 36 89 97 18"
k5_seed_44="52 66 69 89 14"
k5_seed_45="34 53 62 32 10"
k5_seed_46="9 51 5 75 29"
k5_seed_47="45 8 55 70 58"
k5_seed_48="70 40 16 71 91"
k5_seed_49="8 44 52 14 41"
k5_seed_50="63 34 46 81 31"
k5_seed_51="31 64 70 20 29"
k10_seed_42="81 14 3 94 35 31 28 17 13 86"
k10_seed_43="4 36 89 97 18 59 47 85 12 58"
k10_seed_44="52 66 69 89 14 22 48 28 37 3"
k10_seed_45="34 53 62 32 10 38 43 2 9 61"
k10_seed_46="9 51 5 75 29 81 66 19 74 68"
k10_seed_47="45 8 55 70 58 73 43 32 65 49"
k10_seed_48="70 40 16 71 91 68 38 64 24 97"
k10_seed_49="8 44 52 14 41 96 70 65 6 87"
k10_seed_50="63 34 46 81 31 88 60 98 42 10"
k10_seed_51="31 64 70 20 29 32 50 59 93 89"
k20_seed_42="81 14 3 94 35 31 28 17 13 86 69 11 75 54 4 27 29 64 77 71"
k20_seed_43="4 36 89 97 18 59 47 85 12 58 76 63 77 2 65 55 73 79 70 96"
k20_seed_44="52 66 69 89 14 22 48 28 37 3 72 1 12 20 99 65 88 38 81 75"
k20_seed_45="34 53 62 32 10 38 43 2 9 61 1 14 36 15 39 95 83 4 79 8"
k20_seed_46="9 51 5 75 29 81 66 19 74 68 4 3 8 40 80 6 17 70 37 67"
k20_seed_47="45 8 55 70 58 73 43 32 65 49 50 78 5 53 84 3 13 0 33 61"
k20_seed_48="70 40 16 71 91 68 38 64 24 97 55 21 19 84 13 63 83 28 27 65"
k20_seed_49="8 44 52 14 41 96 70 65 6 87 5 97 94 35 19 58 93 80 69 64"
k20_seed_50="63 34 46 81 31 88 60 98 42 10 68 40 28 86 71 19 44 12 24 8"
k20_seed_51="31 64 70 20 29 32 50 59 93 89 67 80 43 75 84 46 49 35 51 0"
k50_seed_42="81 14 3 94 35 31 28 17 13 86 69 11 75 54 4 97 88 27 29 64 77 84 71 25 89 53 93 57 95 0 20 90 43 79 19 82 67 6 5 24 62 22 68 58 38 16 51 2 46 99"
k50_seed_43="4 36 89 18 59 47 85 97 12 58 76 63 77 2 65 55 73 94 79 70 84 50 22 7 13 15 48 19 91 20 93 88 16 49 64 34 23 24 99 3 37 60 11 90 53 10 96 17 61 33"
k50_seed_44="52 66 69 89 14 22 48 28 37 3 92 95 72 1 12 20 65 38 81 93 75 51 41 45 9 43 36 84 88 83 8 85 34 40 13 91 4 94 24 6 30 59 68 57 97 16 47 15 49 90"
k50_seed_45="34 53 62 32 10 38 43 2 9 61 1 14 36 15 39 83 4 79 8 17 99 19 27 54 91 52 18 7 84 22 41 96 45 40 12 74 11 89 48 21 20 50 24 44 47 33 42 82 66 46"
k50_seed_46="9 51 5 75 96 29 81 66 19 74 68 4 3 8 40 80 6 87 17 70 37 67 42 11 50 61 46 91 60 59 99 58 14 55 28 43 56 49 45 93 69 83 38 22 88 57 21 94 25 76"
k50_seed_47="45 8 55 70 58 73 43 32 65 49 50 78 5 53 84 3 13 0 33 61 69 1 40 72 54 29 30 66 91 99 56 94 4 47 96 93 25 15 89 86 46 95 2 98 21 6 23 37 24 92"
k50_seed_48="70 40 16 71 91 68 38 64 24 55 21 19 84 13 63 83 28 27 65 96 81 88 60 82 3 9 54 8 97 85 20 80 7 37 43 79 12 53 15 17 94 41 0 56 36 18 98 72 22 58"
k50_seed_49="8 44 52 14 41 70 65 6 87 5 35 19 58 90 80 69 64 28 77 36 73 47 51 3 31 17 12 54 48 95 45 34 81 92 97 26 99 85 16 68 38 60 20 63 75 96 57 61 37 27"
k50_seed_50="63 34 46 81 31 88 60 42 10 68 40 28 86 71 91 19 44 12 83 89 94 24 8 92 54 11 77 76 55 0 29 43 98 87 13 25 56 17 57 96 37 69 73 62 39 65 67 59 7 26"
k50_seed_51="31 64 70 20 99 29 97 32 50 93 59 90 67 80 43 75 46 49 35 51 0 14 3 68 33 4 61 22 44 38 47 57 24 77 56 21 60 79 58 66 6 52 91 73 69 65 76 37 89 11"

TRAIN_COMMON="--save-epoch 5 --max-epoch 30 -d datasets --data-mode CR --train-batch-size 64 --val-batch-size 16 --version original --epoch best"

# extract_ckpts <out root> <model dir> <experiment> <dataset> <classes> [extra args]
# Checkpoints best and 00 (00 only if it exists), skipping those already extracted.
extract_ckpts () {
    local out=$1 model_dir=$2 exp=$3 dataset=$4 classes=$5; shift 5
    local id=${dataset}_classes_to_remove_${classes// /_}
    for ckpt in best 00; do
        [ "$ckpt" == "00" ] && [ ! -f "${model_dir}/checkpoint_epoch_00.pt" ] && continue
        if [ -f "${out}/${exp}_${id}/${method_name}/original_best/${ckpt}/test.csv" ]; then
            echo "  ${out}: ${exp} ${id} ${ckpt} already extracted"; continue
        fi
        python src/training/main_extractor.py -o ${out} --model-directory models -e ${exp}_${dataset} -m ${exp} -d datasets --dataset ${dataset} --data-mode CR --classes-to-remove ${classes} --unlearning-method ${method_name} --version original_best --checkpoint ${ckpt} "$@"
    done
}
