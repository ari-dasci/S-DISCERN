#!/bin/bash
#SBATCH -c 16
#SBATCH --mem 30g
#SBATCH --job-name unl_miss_multi
#SBATCH --output=slrums/output_%A_%a.out # STDOUT
#SBATCH --gres=gpu:1
#SBATCH --array=[0-2]%3

# Trains the "multiple class removal" (cifar100, seed-based groups) combos
# that are genuinely missing right now, verified against models/ + results/
# AND against the current parser/config code (see extract_missing_multiple.sh
# for why most of what first looked like a multi-class gap turned out to be
# an obsolete naming scheme instead, not a real missing run):
#
#   0: BOUNDARYEXPAND_lr1e-05_embs, efficientnet_v2_s/resnet18/swin_t,
#      the 50-classes-removed groups only (10 seeds each = 30 runs).
#      All other group sizes (5/10/20) already have it; only "50" is missing,
#      consistently across all three architectures.
#   1: EWC_1.0_lambda_MAS_embs_notop, vit_b_16, the 10- and 20-classes-removed
#      groups only (20 runs). Sizes 5 and 50 already have it.
#   2: EWC_10.0_lambda_MAS_embs_notop, vit_b_16, same 10-/20-class groups (20 runs).
#
# Classes-to-remove for each group come from the models/ directory suffix
# (matches the group already used for every other method in that group), and
# --seed is looked up per suffix below so it matches the seed originally used
# to derive that class list in unlearn_multi_5/10/20/50.sh - not just a fixed
# default - since it's the same experimental group as every other method.
#
# Deliberately NOT included (scope decisions already made this session):
#   - The untried SSD_MAS lambda/alpha grid combos (declined for the
#     single-class scope; same call applies here).
#   - EWC "embsunfreeze" / DISTRIBUTIONAL plain "notop": can't be produced
#     with the current code at all (see extract_missing_multiple.sh notes).

eval "$(conda shell.bash hook)"

conda activate unlearning
export PYTHONPATH='.'

# suffix (models/*_cifar100_classes_to_remove_<suffix>) -> seed used to derive it
declare -A group_seed=(
    ["81_14_3_94_35"]=42
    ["4_36_89_97_18"]=43
    ["52_66_69_89_14"]=44
    ["34_53_62_32_10"]=45
    ["9_51_5_75_29"]=46
    ["45_8_55_70_58"]=47
    ["70_40_16_71_91"]=48
    ["8_44_52_14_41"]=49
    ["63_34_46_81_31"]=50
    ["31_64_70_20_29"]=51
    ["81_14_3_94_35_31_28_17_13_86"]=42
    ["4_36_89_97_18_59_47_85_12_58"]=43
    ["52_66_69_89_14_22_48_28_37_3"]=44
    ["34_53_62_32_10_38_43_2_9_61"]=45
    ["9_51_5_75_29_81_66_19_74_68"]=46
    ["45_8_55_70_58_73_43_32_65_49"]=47
    ["70_40_16_71_91_68_38_64_24_97"]=48
    ["8_44_52_14_41_96_70_65_6_87"]=49
    ["63_34_46_81_31_88_60_98_42_10"]=50
    ["31_64_70_20_29_32_50_59_93_89"]=51
    ["81_14_3_94_35_31_28_17_13_86_69_11_75_54_4_27_29_64_77_71"]=42
    ["4_36_89_97_18_59_47_85_12_58_76_63_77_2_65_55_73_79_70_96"]=43
    ["52_66_69_89_14_22_48_28_37_3_72_1_12_20_99_65_88_38_81_75"]=44
    ["34_53_62_32_10_38_43_2_9_61_1_14_36_15_39_95_83_4_79_8"]=45
    ["9_51_5_75_29_81_66_19_74_68_4_3_8_40_80_6_17_70_37_67"]=46
    ["45_8_55_70_58_73_43_32_65_49_50_78_5_53_84_3_13_0_33_61"]=47
    ["70_40_16_71_91_68_38_64_24_97_55_21_19_84_13_63_83_28_27_65"]=48
    ["8_44_52_14_41_96_70_65_6_87_5_97_94_35_19_58_93_80_69_64"]=49
    ["63_34_46_81_31_88_60_98_42_10_68_40_28_86_71_19_44_12_24_8"]=50
    ["31_64_70_20_29_32_50_59_93_89_67_80_43_75_84_46_49_35_51_0"]=51
    ["81_14_3_94_35_31_28_17_13_86_69_11_75_54_4_97_88_27_29_64_77_84_71_25_89_53_93_57_95_0_20_90_43_79_19_82_67_6_5_24_62_22_68_58_38_16_51_2_46_99"]=42
    ["4_36_89_18_59_47_85_97_12_58_76_63_77_2_65_55_73_94_79_70_84_50_22_7_13_15_48_19_91_20_93_88_16_49_64_34_23_24_99_3_37_60_11_90_53_10_96_17_61_33"]=43
    ["52_66_69_89_14_22_48_28_37_3_92_95_72_1_12_20_65_38_81_93_75_51_41_45_9_43_36_84_88_83_8_85_34_40_13_91_4_94_24_6_30_59_68_57_97_16_47_15_49_90"]=44
    ["34_53_62_32_10_38_43_2_9_61_1_14_36_15_39_83_4_79_8_17_99_19_27_54_91_52_18_7_84_22_41_96_45_40_12_74_11_89_48_21_20_50_24_44_47_33_42_82_66_46"]=45
    ["9_51_5_75_96_29_81_66_19_74_68_4_3_8_40_80_6_87_17_70_37_67_42_11_50_61_46_91_60_59_99_58_14_55_28_43_56_49_45_93_69_83_38_22_88_57_21_94_25_76"]=46
    ["45_8_55_70_58_73_43_32_65_49_50_78_5_53_84_3_13_0_33_61_69_1_40_72_54_29_30_66_91_99_56_94_4_47_96_93_25_15_89_86_46_95_2_98_21_6_23_37_24_92"]=47
    ["70_40_16_71_91_68_38_64_24_55_21_19_84_13_63_83_28_27_65_96_81_88_60_82_3_9_54_8_97_85_20_80_7_37_43_79_12_53_15_17_94_41_0_56_36_18_98_72_22_58"]=48
    ["8_44_52_14_41_70_65_6_87_5_35_19_58_90_80_69_64_28_77_36_73_47_51_3_31_17_12_54_48_95_45_34_81_92_97_26_99_85_16_68_38_60_20_63_75_96_57_61_37_27"]=49
    ["63_34_46_81_31_88_60_42_10_68_40_28_86_71_91_19_44_12_83_89_94_24_8_92_54_11_77_76_55_0_29_43_98_87_13_25_56_17_57_96_37_69_73_62_39_65_67_59_7_26"]=50
    ["31_64_70_20_99_29_97_32_50_93_59_90_67_80_43_75_46_49_35_51_0_14_3_68_33_4_61_22_44_38_47_57_24_77_56_21_60_79_58_66_6_52_91_73_69_65_76_37_89_11"]=51
)

case "$SLURM_ARRAY_TASK_ID" in
    0)
        experiments=("efficientnet_v2_s" "resnet18" "swin_t")
        target_sizes="50"
        method_name="BOUNDARYEXPAND_lr1e-05_embs"
        unlearn_args="--unlearning-method BOUNDARYEXPAND --freeze embs"
        ;;
    1)
        experiments=("vit_b_16")
        target_sizes="10 20"
        method_name="EWC_1.0_lambda_MAS_embs_notop"
        unlearn_args="--unlearning-method EWC --loss-forget notop --lambda-ewc 1.0 --ewc-coefficients MAS --emb-unfreeze embs"
        ;;
    2)
        experiments=("vit_b_16")
        target_sizes="10 20"
        method_name="EWC_10.0_lambda_MAS_embs_notop"
        unlearn_args="--unlearning-method EWC --loss-forget notop --lambda-ewc 10.0 --ewc-coefficients MAS --emb-unfreeze embs"
        ;;
esac

dataset="cifar100"

for experiment in "${experiments[@]}"; do
    prefix="models/${experiment}_${dataset}_classes_to_remove_"
    for model_dir in ${prefix}*_*; do
        [ -d "$model_dir" ] || continue
        suffix="${model_dir#$prefix}"

        size=$(tr '_' ' ' <<< "$suffix" | wc -w)
        if [[ ! " $target_sizes " == *" $size "* ]]; then
            continue
        fi

        seed="${group_seed[$suffix]}"
        if [ -z "$seed" ]; then
            echo "WARNING: no known seed for classes_to_remove suffix '${suffix}', skipping"
            continue
        fi
        classes_to_remove="${suffix//_/ }"

        ckpt="${model_dir}/${method_name}/original_best/checkpoint_best.pt"
        if [ -f "$ckpt" ]; then
            echo "Skipping ${experiment} ${dataset} classes [${classes_to_remove}] (already trained)"
            continue
        fi

        echo "Starting unlearning process for ${experiment} of ${dataset} classes [${classes_to_remove}] (seed ${seed}) with method ${method_name}"
        python src/unlearning/main_unlearning.py -o models -e ${experiment}_${dataset} -m ${experiment} --save-epoch 5 --max-epoch 30 -d datasets --dataset ${dataset} --data-mode CR --train-batch-size 64 --val-batch-size 16 --seed ${seed} --classes-to-remove ${classes_to_remove} --version original --epoch best ${unlearn_args}
    done
done

conda deactivate
