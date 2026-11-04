#!/bin/bash 
#SBATCH -c 16
#SBATCH --mem 20g
#SBATCH --job-name extract
#SBATC#H -o ./slrums/%j.out # STDOUT
#SBATCH --output=slrums/output_%A_%a.out # STDOUT
#SBATCH --gres=gpu:1
#SBATCH --array=0-16%3


eval "$(conda shell.bash hook)"

conda activate unlearning
export PYTHONPATH='.'
# methods=("BADTEACHER_T1.0_forgetOnly" 
# "BADTEACHER_T1.0_withRetain" 
# "DISTRIBUTIONAL_1000_standard_clustering_cov_diagonal_syntetic_fixed_freeze_forget_confidence_penalty" 
# "DISTRIBUTIONAL_1000_standard_cov_diagonal_syntetic_fixed_freeze_forget_confidence_penalty" 
# "FISHERFORGETTING_noise1.0_alpha1e-06_moderetain" 
# "FISHERFORGETTING_noise1.0_alpha1e-06_modesubtract" 
# "NOISY_1000_standard_clustering_cov_diagonal_syntetic_fixed_freeze_forget_confidence_penalty" 
# "NOISY_1000_standard_clustering_cov_lasso_syntetic_fixed_freeze_forget_confidence_penalty" 
# "NOISY_1000_standard_cov_diagonal_syntetic_fixed_freeze_forget_confidence_penalty" 
# "NOISY_1000_standard_cov_lasso_syntetic_fixed_freeze_forget_confidence_penalty" 
# "RANDOMLABEL_noret" 
# "SSD_FISHER_lambda_1.0_alpha_10.0" 
# "SSD_FISHER_lambda_1.0_alpha_5.0" 
# "SSD_MAS_lambda_1.0_alpha_10.0" 
# "SSD_MAS_lambda_1.0_alpha_5.0" )

methods=(
    "BADTEACHER_T1.0_forgetOnly_embs"
    "BOUNDARYEXPAND_lr1e-05_embs"
    "BOUNDARYSHRINK_eps0.1_lr1e-05_embs"
    # "DISTRIBUTIONAL_1000_standard_cov_diagonal_syntetic_fixed_freeze_forget_notop"
    # "DISTRIBUTIONAL_1000_standard_clustering_cov_diagonal_raro_freeze_forget_confidence_penalty"
    # "DISTRIBUTIONAL_1000_standard_cov_diagonal_syntetic_fixed_freeze_forget_confidence_penalty"
    # "DISTRIBUTIONAL_1000_standard_cov_diagonal_syntetic_raro_forget_confidence_penalty"
    # "EWC_10.0_lambda_MAS_freeze_confidence_penalty"
    # "EWC_1.0_lambda_MAS_freeze_confidence_penalty"
    # "ISPF_nz256_T1.0_freeze"
    # "NOISY_1000_standard_clustering_cov_diagonal_raro_freeze_forget_confidence_penalty"
    # "NOISY_1000_standard_clustering_cov_diagonal_syntetic_fixed_freeze_forget_confidence_penalty"
    # "NOISY_1000_standard_cov_diagonal_raro_freeze_forget_confidence_penalty"
    # "NOISY_1000_standard_cov_diagonal_syntetic_fixed_freeze_forget_confidence_penalty"
    "RANDOMLABEL_noret_embs"
    # "RANDOMLABEL_ret0.1"
    "SSD_MAS_lambda_1.0_alpha_10.0_embs"
    "SSD_MAS_lambda_1.0_alpha_5.0_embs"
    # "RETRAIN"
    # "DISTRIBUTIONAL_1000_standard_cov_diagonal_syntetic_fixed_embs_forget_notop"

    "FISHERFORGETTING_noise1.0_alpha1e-06_modesubtract_embs"

    "DELETE_embs"

    "SFU_damping0.0001_scale1.0_hessforget_embs"

    "NOISY_1000_standard_clustering_cov_ledoit_wolf_syntetic_fixed_freeze_forget_notop"

    "NOISY_1000_standard_clustering_cov_ledoit_wolf_syntetic_fixed_embs_forget_notop"

    "RANDOMLABEL_noret_freeze_forbidden"
    "RANDOMLABEL_noret_embs_forbidden"

)


experiments=("swin_t" "efficientnet_v2_s" "resnet18") # "swin_t" "efficientnet_v2_s" "vit_b_16" "resnet18"
datasets=("cifar100")
seeds=(42 43 44 45 46 47 48 49 50 51)


methods=(

    "DISTRIBUTIONAL_1000_standard_cov_ledoit_wolf_syntetic_fixed_embs_forget_notop" 
    "DISTRIBUTIONAL_1000_standard_cov_ledoit_wolf_syntetic_moving_embs_forget_notop" 
    "NOISY_1000_standard_cov_ledoit_wolf_syntetic_fixed_embs_forget_notop"
    "NOISY_1000_standard_cov_ledoit_wolf_syntetic_fixed_freeze_forget_notop"
)

methods=(


    "--unlearning-method EWC --loss-forget notop --lambda-ewc 1.0 --ewc-coefficients MAS --emb-unfreeze embs"
    "--unlearning-method EWC --loss-forget notop --lambda-ewc 10.0 --ewc-coefficients MAS --emb-unfreeze embs"

    "--unlearning-method EWC --loss-forget confidence_penalty --lambda-ewc 1.0 --ewc-coefficients MAS --emb-unfreeze embs"
    "--unlearning-method EWC --loss-forget confidence_penalty --lambda-ewc 10.0 --ewc-coefficients MAS --emb-unfreeze embs"

    "--unlearning-method EWC --loss-forget confidence_penalty --lambda-ewc 1.0 --ewc-coefficients MAS --emb-unfreeze freeze"
    "--unlearning-method EWC --loss-forget confidence_penalty --lambda-ewc 10.0 --ewc-coefficients MAS --emb-unfreeze freeze"
    
)

methods=(
    "--unlearning-method DISTRIBUTIONAL --loss-forget corrected_confidence_penalty --syntetic-samples 1000 --fixed-syntetic --emb-unfreeze embs --covariance-type ledoit_wolf"
)

methods=(
    "--unlearning-method NOISY --loss-forget confidence_penalty --syntetic-samples 1000 --fixed-syntetic --emb-unfreeze embs --covariance-type ledoit_wolf"
    "--unlearning-method NOISY --loss-forget notop --syntetic-samples 1000 --fixed-syntetic --emb-unfreeze embs --covariance-type ledoit_wolf"
)

methods=(
    "BOUNDARYSHRINK_eps0.1_lr1e-05_embs"
    "NOISY_1000_standard_cov_ledoit_wolf_syntetic_fixed_embs_forget_confidence_penalty"
    "NOISY_1000_standard_cov_ledoit_wolf_syntetic_fixed_embs_forget_notop"
    "DISTRIBUTIONAL_1000_standard_cov_ledoit_wolf_syntetic_fixed_embs_forget_corrected_confidence_penalty"
    
    "EWC_10.0_lambda_MAS_embs_notop"
    "EWC_1.0_lambda_MAS_embs_notop"
    "EWC_10.0_lambda_MAS_embs_confidence_penalty"
    "EWC_1.0_lambda_MAS_embs_confidence_penalty"
    "EWC_10.0_lambda_MAS_embs_confidence_penalty"
    "EWC_1.0_lambda_MAS_embs_confidence_penalty"

    "EWC_10.0_lambda_MAS_embsunfreeze_notop"
    "EWC_1.0_lambda_MAS_embsunfreeze_notop"
    "EWC_10.0_lambda_MAS_embsunfreeze_confidence_penalty"
    "EWC_1.0_lambda_MAS_embsunfreeze_confidence_penalty"
    "EWC_10.0_lambda_MAS_embsunfreeze_confidence_penalty"
    "EWC_1.0_lambda_MAS_embsunfreeze_confidence_penalty"
    
)


methods=(

    "BADTEACHER_T1.0_forgetOnly_embs"
    "BOUNDARYEXPAND_lr1e-05_embs"
    "DELETE_embs"
    "BOUNDARYSHRINK_eps0.1_lr1e-05_embs"


    "NOISY_1000_standard_cov_ledoit_wolf_syntetic_fixed_embs_forget_confidence_penalty"
    "NOISY_1000_standard_cov_ledoit_wolf_syntetic_fixed_embs_forget_notop"
    "DISTRIBUTIONAL_1000_standard_cov_ledoit_wolf_syntetic_fixed_embs_forget_corrected_confidence_penalty"
    "DISTRIBUTIONAL_1000_standard_cov_ledoit_wolf_syntetic_fixed_embs_forget_notop"
    

    "EWC_10.0_lambda_MAS_embs_notop"
    "EWC_1.0_lambda_MAS_embs_notop"
    "RANDOMLABEL_noret_embs_forbidden"

    "EWC_10.0_lambda_MAS_embsunfreeze_notop"
    "EWC_1.0_lambda_MAS_embsunfreeze_notop"

    "SSD_MAS_lambda_1.0_alpha_10.0_embs"
    "SSD_MAS_lambda_0.1_alpha_1.0_embs"
    "SSD_MAS_lambda_1.0_alpha_5.0_embs"

    "RETRAIN"
    
)

method=${methods[$SLURM_ARRAY_TASK_ID]}

for experiment in "${experiments[@]}"; do

    seed_42="81 14 3 94 35"
    seed_43="4 36 89 97 18"
    seed_44="52 66 69 89 14"
    seed_45="34 53 62 32 10"
    seed_46="9 51 5 75 29"
    seed_47="45 8 55 70 58"
    seed_48="70 40 16 71 91"
    seed_49="8 44 52 14 41"
    seed_50="63 34 46 81 31"
    seed_51="31 64 70 20 29"
    for dataset in "${datasets[@]}"; do
        for seed in "${seeds[@]}"; do
            classes_to_remove_var="seed_${seed}[@]"
            classes_to_remove=("${!classes_to_remove_var}")
            classes_to_remove_coneted="${classes_to_remove// /_}"
            # for epoch2 in $(ls models/${experiment}_${dataset}_classes_to_remove_${classes_to_remove_coneted}/${method}/original_best/checkpoint_epoch_*.pt | sed -E 's/.*checkpoint_epoch_([0-9]+)\.pt/\1/'); do
            #     echo "Starting extraction process for ${experiment} epc ${epoch2} on ${dataset} original_best with method ${method}"
            #     python src/training/main_extractor.py -o results --model-directory models -e ${experiment}_${dataset} -m ${experiment} -d datasets --dataset ${dataset} --data-mode CR --classes-to-remove ${classes_to_remove} --unlearning-method ${method} --version original_best --checkpoint ${epoch2}
            # done

            echo "Starting extraction process for ${experiment} epc best on ${dataset} original_best with method ${method}"
            python src/training/main_extractor.py -o results --model-directory models -e ${experiment}_${dataset} -m ${experiment} -d datasets --dataset ${dataset} --data-mode CR --classes-to-remove ${classes_to_remove} --unlearning-method ${method} --version original_best --checkpoint best
            echo "Starting extraction process for ${experiment} epc 00 on ${dataset} original_best with method ${method}"
            python src/training/main_extractor.py -o results --model-directory models -e ${experiment}_${dataset} -m ${experiment} -d datasets --dataset ${dataset} --data-mode CR --classes-to-remove ${classes_to_remove} --unlearning-method ${method} --version original_best --checkpoint "00"
        done
    done

    seed_42="81 14 3 94 35 31 28 17 13 86"
    seed_43="4 36 89 97 18 59 47 85 12 58"
    seed_44="52 66 69 89 14 22 48 28 37 3"
    seed_45="34 53 62 32 10 38 43 2 9 61"
    seed_46="9 51 5 75 29 81 66 19 74 68"
    seed_47="45 8 55 70 58 73 43 32 65 49"
    seed_48="70 40 16 71 91 68 38 64 24 97"
    seed_49="8 44 52 14 41 96 70 65 6 87"
    seed_50="63 34 46 81 31 88 60 98 42 10"
    seed_51="31 64 70 20 29 32 50 59 93 89"
    for dataset in "${datasets[@]}"; do
        for seed in "${seeds[@]}"; do
            classes_to_remove_var="seed_${seed}[@]"
            classes_to_remove=("${!classes_to_remove_var}")
            classes_to_remove_coneted="${classes_to_remove// /_}"
            # for epoch2 in $(ls models/${experiment}_${dataset}_classes_to_remove_${classes_to_remove_coneted}/${method}/original_best/checkpoint_epoch_*.pt | sed -E 's/.*checkpoint_epoch_([0-9]+)\.pt/\1/'); do
            #     echo "Starting extraction process for ${experiment} epc ${epoch2} on ${dataset} original_best with method ${method}"
            #     python src/training/main_extractor.py -o results --model-directory models -e ${experiment}_${dataset} -m ${experiment} -d datasets --dataset ${dataset} --data-mode CR --classes-to-remove ${classes_to_remove} --unlearning-method ${method} --version original_best --checkpoint ${epoch2}
            # done

            echo "Starting extraction process for ${experiment} epc best on ${dataset} original_best with method ${method}"
            python src/training/main_extractor.py -o results --model-directory models -e ${experiment}_${dataset} -m ${experiment} -d datasets --dataset ${dataset} --data-mode CR --classes-to-remove ${classes_to_remove} --unlearning-method ${method} --version original_best --checkpoint best
            echo "Starting extraction process for ${experiment} epc 00 on ${dataset} original_best with method ${method}"
            python src/training/main_extractor.py -o results --model-directory models -e ${experiment}_${dataset} -m ${experiment} -d datasets --dataset ${dataset} --data-mode CR --classes-to-remove ${classes_to_remove} --unlearning-method ${method} --version original_best --checkpoint "00"
        done
    done

    seed_42="81 14 3 94 35 31 28 17 13 86 69 11 75 54 4 27 29 64 77 71"
    seed_43="4 36 89 97 18 59 47 85 12 58 76 63 77 2 65 55 73 79 70 96"
    seed_44="52 66 69 89 14 22 48 28 37 3 72 1 12 20 99 65 88 38 81 75"
    seed_45="34 53 62 32 10 38 43 2 9 61 1 14 36 15 39 95 83 4 79 8"
    seed_46="9 51 5 75 29 81 66 19 74 68 4 3 8 40 80 6 17 70 37 67" 
    seed_47="45 8 55 70 58 73 43 32 65 49 50 78 5 53 84 3 13 0 33 61"
    seed_48="70 40 16 71 91 68 38 64 24 97 55 21 19 84 13 63 83 28 27 65"
    seed_49="8 44 52 14 41 96 70 65 6 87 5 97 94 35 19 58 93 80 69 64"
    seed_50="63 34 46 81 31 88 60 98 42 10 68 40 28 86 71 19 44 12 24 8"
    seed_51="31 64 70 20 29 32 50 59 93 89 67 80 43 75 84 46 49 35 51 0"

    for dataset in "${datasets[@]}"; do
        for seed in "${seeds[@]}"; do
            classes_to_remove_var="seed_${seed}[@]"
            classes_to_remove=("${!classes_to_remove_var}")
            classes_to_remove_coneted="${classes_to_remove// /_}"
            # for epoch2 in $(ls models/${experiment}_${dataset}_classes_to_remove_${classes_to_remove_coneted}/${method}/original_best/checkpoint_epoch_*.pt | sed -E 's/.*checkpoint_epoch_([0-9]+)\.pt/\1/'); do
            #     echo "Starting extraction process for ${experiment} epc ${epoch2} on ${dataset} original_best with method ${method}"
            #     python src/training/main_extractor.py -o results --model-directory models -e ${experiment}_${dataset} -m ${experiment} -d datasets --dataset ${dataset} --data-mode CR --classes-to-remove ${classes_to_remove} --unlearning-method ${method} --version original_best --checkpoint ${epoch2}
            # done

            echo "Starting extraction process for ${experiment} epc best on ${dataset} original_best with method ${method}"
            python src/training/main_extractor.py -o results --model-directory models -e ${experiment}_${dataset} -m ${experiment} -d datasets --dataset ${dataset} --data-mode CR --classes-to-remove ${classes_to_remove} --unlearning-method ${method} --version original_best --checkpoint best
            echo "Starting extraction process for ${experiment} epc 00 on ${dataset} original_best with method ${method}"
            python src/training/main_extractor.py -o results --model-directory models -e ${experiment}_${dataset} -m ${experiment} -d datasets --dataset ${dataset} --data-mode CR --classes-to-remove ${classes_to_remove} --unlearning-method ${method} --version original_best --checkpoint "00"
        done
    done

    seed_42="81 14 3 94 35 31 28 17 13 86 69 11 75 54 4 97 88 27 29 64 77 84 71 25 89 53 93 57 95 0 20 90 43 79 19 82 67 6 5 24 62 22 68 58 38 16 51 2 46 99"
    seed_43="4 36 89 18 59 47 85 97 12 58 76 63 77 2 65 55 73 94 79 70 84 50 22 7 13 15 48 19 91 20 93 88 16 49 64 34 23 24 99 3 37 60 11 90 53 10 96 17 61 33"
    seed_44="52 66 69 89 14 22 48 28 37 3 92 95 72 1 12 20 65 38 81 93 75 51 41 45 9 43 36 84 88 83 8 85 34 40 13 91 4 94 24 6 30 59 68 57 97 16 47 15 49 90"
    seed_45="34 53 62 32 10 38 43 2 9 61 1 14 36 15 39 83 4 79 8 17 99 19 27 54 91 52 18 7 84 22 41 96 45 40 12 74 11 89 48 21 20 50 24 44 47 33 42 82 66 46"
    seed_46="9 51 5 75 96 29 81 66 19 74 68 4 3 8 40 80 6 87 17 70 37 67 42 11 50 61 46 91 60 59 99 58 14 55 28 43 56 49 45 93 69 83 38 22 88 57 21 94 25 76"
    seed_47="45 8 55 70 58 73 43 32 65 49 50 78 5 53 84 3 13 0 33 61 69 1 40 72 54 29 30 66 91 99 56 94 4 47 96 93 25 15 89 86 46 95 2 98 21 6 23 37 24 92"
    seed_48="70 40 16 71 91 68 38 64 24 55 21 19 84 13 63 83 28 27 65 96 81 88 60 82 3 9 54 8 97 85 20 80 7 37 43 79 12 53 15 17 94 41 0 56 36 18 98 72 22 58"
    seed_49="8 44 52 14 41 70 65 6 87 5 35 19 58 90 80 69 64 28 77 36 73 47 51 3 31 17 12 54 48 95 45 34 81 92 97 26 99 85 16 68 38 60 20 63 75 96 57 61 37 27"
    seed_50="63 34 46 81 31 88 60 42 10 68 40 28 86 71 91 19 44 12 83 89 94 24 8 92 54 11 77 76 55 0 29 43 98 87 13 25 56 17 57 96 37 69 73 62 39 65 67 59 7 26"
    seed_51="31 64 70 20 99 29 97 32 50 93 59 90 67 80 43 75 46 49 35 51 0 14 3 68 33 4 61 22 44 38 47 57 24 77 56 21 60 79 58 66 6 52 91 73 69 65 76 37 89 11"

    for dataset in "${datasets[@]}"; do
        for seed in "${seeds[@]}"; do
            classes_to_remove_var="seed_${seed}[@]"
            classes_to_remove=("${!classes_to_remove_var}")
            classes_to_remove_coneted="${classes_to_remove// /_}"
            # for epoch2 in $(ls models/${experiment}_${dataset}_classes_to_remove_${classes_to_remove_coneted}/${method}/original_best/checkpoint_epoch_*.pt | sed -E 's/.*checkpoint_epoch_([0-9]+)\.pt/\1/'); do
            #     echo "Starting extraction process for ${experiment} epc ${epoch2} on ${dataset} original_best with method ${method}"
            #     python src/training/main_extractor.py -o results --model-directory models -e ${experiment}_${dataset} -m ${experiment} -d datasets --dataset ${dataset} --data-mode CR --classes-to-remove ${classes_to_remove} --unlearning-method ${method} --version original_best --checkpoint ${epoch2}
            # done

            echo "Starting extraction process for ${experiment} epc best on ${dataset} original_best with method ${method}"
            python src/training/main_extractor.py -o results --model-directory models -e ${experiment}_${dataset} -m ${experiment} -d datasets --dataset ${dataset} --data-mode CR --classes-to-remove ${classes_to_remove} --unlearning-method ${method} --version original_best --checkpoint best
            echo "Starting extraction process for ${experiment} epc 00 on ${dataset} original_best with method ${method}"
            python src/training/main_extractor.py -o results --model-directory models -e ${experiment}_${dataset} -m ${experiment} -d datasets --dataset ${dataset} --data-mode CR --classes-to-remove ${classes_to_remove} --unlearning-method ${method} --version original_best --checkpoint "00"
        done
    done

done
conda deactivate
