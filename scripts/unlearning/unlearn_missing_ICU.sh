#!/bin/bash
#SBATCH -c 16
#SBATCH --mem 30g
#SBATCH --job-name unl_miss_icu
#SBATCH --output=slrums/output_%A_%a.out # STDOUT
#SBATCH --gres=gpu:1
#SBATCH --array=[3]%3

# Trains the ICU (incremental/continual class unlearning) combos that are
# genuinely missing from results_ICU/, restricted to the methods tracked in
# src/plots/plot_figures.py's FIXED_METHODS_ICU. Verified against results_ICU/ directly
# (not models/ - ICU checkpoints get pruned once unlearning_results_ICU.csv
# is written, same pattern as single/multi), so "missing" here means the CSV
# for that method+group truly doesn't exist yet.
#
# Deliberately does NOT try to carve out an exact non-overlapping scope
# against the two currently-held jobs (169329: DISTRIBUTIONAL_notop variants
# on cifar100, all 4 archs; 169365: RETRAIN on swin_t/cifar100 only) - held
# back on purpose while single/multi finish first. Instead every task below
# just skips a (arch, dataset, group) combo when its
# unlearning_results_ICU.csv already exists, so running this before OR after
# those held jobs are released is equally safe: whichever runs second simply
# finds nothing left to do for the overlapping cells.
#
#   0: RETRAIN - all 4 archs x cifar10/mnist/cifar100. Currently missing for
#      efficientnet_v2_s and resnet18 everywhere, and for swin_t on
#      cifar10/mnist (vit_b_16, and swin_t/cifar100 once 169365 runs, already
#      covered - skipped automatically).
#   1: DISTRIBUTIONAL_..._forget_notop - all 4 archs x cifar10/mnist/cifar100.
#      Currently missing for efficientnet_v2_s/resnet18/swin_t on
#      cifar10/mnist (vit_b_16 already has it everywhere; cifar100 for the
#      other 3 archs will be covered once 169329 runs - skipped automatically).
#   2: DISTRIBUTIONAL_..._forget_corrected_confidence_penalty - all 4 archs x
#      cifar10/mnist/cifar100. Not addressed by any currently queued job.
#      Missing on cifar10/mnist for all 4 archs, and on cifar100 for vit_b_16
#      only (efficientnet_v2_s/resnet18/swin_t already have it there).
#   3: NOISY_..._forget_notop, vit_b_16, cifar100 only - 3 of the 10 seed
#      groups never finished (seeds 42, 44, 45).
#
# Deliberately excluded (same call as for single/multi): the untried
# SSD_MAS/EWC hyperparameter grid, and per-architecture SSD/EWC "winners"
# (src/plots/plot_figures.py reuses the single-class winners for those, it doesn't
# re-sweep them for ICU).

eval "$(conda shell.bash hook)"

conda activate unlearning
export PYTHONPATH='.'

# classes_to_remove (underscore form) -> seed used to derive it.
# Shared by cifar10/mnist (same 10 five-class permutation groups) and, for
# the longer 20-class strings, cifar100 (same seed groups as the "multiple" scope).
declare -A group_seed=(
    ["1_0_4_9_6"]=42
    ["0_4_2_3_7"]=43
    ["6_8_1_7_3"]=44
    ["4_6_7_2_0"]=45
    ["1_6_0_4_8"]=46
    ["5_1_6_4_3"]=47
    ["8_5_2_4_9"]=48
    ["1_5_6_0_2"]=49
    ["7_4_5_9_1"]=50
    ["3_8_2_1_6"]=51

    ["31_64_70_20_29_32_50_59_93_89_67_80_43_75_84_46_49_35_51_0"]=42
    ["4_36_89_97_18_59_47_85_12_58_76_63_77_2_65_55_73_79_70_96"]=43
    ["52_66_69_89_14_22_48_28_37_3_72_1_12_20_99_65_88_38_81_75"]=44
    ["34_53_62_32_10_38_43_2_9_61_1_14_36_15_39_95_83_4_79_8"]=45
    ["9_51_5_75_29_81_66_19_74_68_4_3_8_40_80_6_17_70_37_67"]=46
    ["45_8_55_70_58_73_43_32_65_49_50_78_5_53_84_3_13_0_33_61"]=47
    ["70_40_16_71_91_68_38_64_24_97_55_21_19_84_13_63_83_28_27_65"]=48
    ["81_14_3_94_35_31_28_17_13_86_69_11_75_54_4_27_29_64_77_71"]=49
    ["8_44_52_14_41_96_70_65_6_87_5_97_94_35_19_58_93_80_69_64"]=50
    ["63_34_46_81_31_88_60_98_42_10_68_40_28_86_71_19_44_12_24_8"]=51
)

declare -A groups
groups["cifar10"]="1_0_4_9_6 0_4_2_3_7 6_8_1_7_3 4_6_7_2_0 1_6_0_4_8 5_1_6_4_3 8_5_2_4_9 1_5_6_0_2 7_4_5_9_1 3_8_2_1_6"
groups["mnist"]="1_0_4_9_6 0_4_2_3_7 6_8_1_7_3 4_6_7_2_0 1_6_0_4_8 5_1_6_4_3 8_5_2_4_9 1_5_6_0_2 7_4_5_9_1 3_8_2_1_6"
groups["cifar100"]="31_64_70_20_29_32_50_59_93_89_67_80_43_75_84_46_49_35_51_0 4_36_89_97_18_59_47_85_12_58_76_63_77_2_65_55_73_79_70_96 52_66_69_89_14_22_48_28_37_3_72_1_12_20_99_65_88_38_81_75 34_53_62_32_10_38_43_2_9_61_1_14_36_15_39_95_83_4_79_8 9_51_5_75_29_81_66_19_74_68_4_3_8_40_80_6_17_70_37_67 45_8_55_70_58_73_43_32_65_49_50_78_5_53_84_3_13_0_33_61 70_40_16_71_91_68_38_64_24_97_55_21_19_84_13_63_83_28_27_65 81_14_3_94_35_31_28_17_13_86_69_11_75_54_4_27_29_64_77_71 8_44_52_14_41_96_70_65_6_87_5_97_94_35_19_58_93_80_69_64 63_34_46_81_31_88_60_98_42_10_68_40_28_86_71_19_44_12_24_8"

case "$SLURM_ARRAY_TASK_ID" in
    0)
        # RETRAIN. swin_t/cifar100 is deliberately left out - that's exactly
        # what the held job 169365 (unlearn_ICU_20_RETRAIN.sh) will do.
        experiment_datasets=(
            "efficientnet_v2_s:cifar100"
            "resnet18:cifar100"
        )
        method_name="RETRAIN"
        unlearn_args="--unlearning-method RETRAIN"
        ;;
    1)
        # DISTRIBUTIONAL notop. cifar100 for these 3 archs is deliberately
        # left out - that's exactly what the held job 169329
        # (unlearn_ICU_20.sh, array task 0) will do; vit_b_16 already has it
        # everywhere so it's not listed at all.
        experiment_datasets=(
            "efficientnet_v2_s:cifar100"
            "resnet18:cifar100"
            "swin_t:cifar100"
        )
        method_name="DISTRIBUTIONAL_1000_standard_cov_ledoit_wolf_syntetic_fixed_embs_forget_notop"
        unlearn_args="--unlearning-method DISTRIBUTIONAL --loss-forget notop --syntetic-samples 1000 --fixed-syntetic --emb-unfreeze embs --covariance-type ledoit_wolf"
        ;;
    2)
        # DISTRIBUTIONAL corrected_confidence_penalty. Not addressed by any
        # currently queued/held job, so every arch is in scope; the
        # already-done cells (efficientnet_v2_s/resnet18/swin_t on cifar100)
        # are skipped by the results_ICU check below, not by exclusion here.
        experiment_datasets=(
            "efficientnet_v2_s:cifar10 mnist cifar100"
            "resnet18:cifar10 mnist cifar100"
            "swin_t:cifar10 mnist cifar100"
            "vit_b_16:cifar10 mnist cifar100"
        )
        method_name="DISTRIBUTIONAL_1000_standard_cov_ledoit_wolf_syntetic_fixed_embs_forget_corrected_confidence_penalty"
        unlearn_args="--unlearning-method DISTRIBUTIONAL --loss-forget corrected_confidence_penalty --syntetic-samples 1000 --fixed-syntetic --emb-unfreeze embs --covariance-type ledoit_wolf"
        ;;
    3)
        experiment_datasets=(
            "vit_b_16:cifar100"
        )
        method_name="NOISY_1000_standard_cov_ledoit_wolf_syntetic_fixed_embs_forget_notop"
        unlearn_args="--unlearning-method NOISY --loss-forget notop --syntetic-samples 1000 --fixed-syntetic --emb-unfreeze embs --covariance-type ledoit_wolf"
        ;;
esac

for entry in "${experiment_datasets[@]}"; do
    experiment="${entry%%:*}"
    datasets="${entry#*:}"
    for dataset in $datasets; do
        for suffix in ${groups[$dataset]}; do
            seed="${group_seed[$suffix]}"
            if [ -z "$seed" ]; then
                echo "WARNING: no known seed for classes_to_remove suffix '${suffix}', skipping"
                continue
            fi
            classes_to_remove="${suffix//_/ }"

            # A group is done only if its CSV has a row for every step (one
            # step per class to forget). An existing but header-only/partial
            # file (crashed run) must NOT count as done. Same criterion as the
            # early skip in main_unlearning_ICU.py (steps 1..N all present).
            rescsv="results_ICU/${experiment}_${dataset}_classes_to_remove_${suffix}/${method_name}/original_best/unlearning_results_ICU.csv"
            n_steps=$(wc -w <<< "$classes_to_remove")
            steps_done=0
            if [ -f "$rescsv" ]; then
                steps_done=$(awk -F',' 'NR>1 && $4 ~ /^[0-9]+$/ {s[$4]=1} END{n=0; for(k in s) n++; print n}' "$rescsv")
            fi
            if [ "$steps_done" -ge "$n_steps" ]; then
                echo "Skipping ${experiment} ${dataset} classes [${classes_to_remove}] (already in results_ICU, ${steps_done}/${n_steps} steps)"
                continue
            fi
            if [ -f "$rescsv" ]; then
                echo "Resuming ${experiment} ${dataset} classes [${classes_to_remove}]: existing CSV only has ${steps_done}/${n_steps} steps"
            fi

            echo "Starting incremental class unlearning process for ${experiment} of ${dataset} classes [${classes_to_remove}] (seed ${seed}) with method ${method_name}"
            python src/unlearning/main_unlearning_ICU.py -o models -e ${experiment}_${dataset} -m ${experiment} --save-epoch 5 --max-epoch 30 -d datasets --dataset ${dataset} --data-mode CR --train-batch-size 64 --val-batch-size 16 --seed ${seed} --classes-to-remove ${classes_to_remove} --version original --epoch best ${unlearn_args}
        done
    done
done

conda deactivate
