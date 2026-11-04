#!/bin/bash 
#SBATCH -c 16
#SBATCH --mem 20g
#SBATCH --job-name extract
#SBATC#H -o ./slrums/%j.out # STDOUT
#SBATCH --output=slrums/output_%A_%a.out # STDOUT
#SBATCH --gres=gpu:1
#SBATCH --array=0-3%3

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
    # "BADTEACHER_T1.0_forgetOnly"
    
    
    # "DISTRIBUTIONAL_1000_standard_clustering_cov_diagonal_syntetic_fixed_freeze_forget_confidence_penalty"
    
    # "DISTRIBUTIONAL_1000_standard_cov_diagonal_syntetic_fixed_freeze_forget_confidence_penalty"
    
    "EWC_10.0_lambda_MAS_freeze_notop"
    "EWC_1.0_lambda_MAS_freeze_notop"
    # "ISPF_nz256_T1.02"
    
    # "NOISY_1000_standard_clustering_cov_diagonal_syntetic_fixed_freeze_forget_confidence_penalty"
    
    "NOISY_1000_standard_cov_ledoit_wolf_syntetic_fixed_freeze_forget_notop"
    # "RANDOMLABEL_noret"
    # "RANDOMLABEL_ret0.1"
    # "SSD_MAS_lambda_1.0_alpha_10.0"
    # "SSD_MAS_lambda_1.0_alpha_5.0"

    # "DISTRIBUTIONAL_1000_standard_cov_diagonal_syntetic_fixed_freeze_forget_notop"
    # "DISTRIBUTIONAL_1000_standard_clustering_cov_diagonal_syntetic_fixed_freeze_forget_notop"


    # "DISTRIBUTIONAL_1000_standard_cov_diagonal_syntetic_fixed_freeze_self_forget_1000_forget_notop"
    # "DISTRIBUTIONAL_1000_standard_clustering_cov_diagonal_syntetic_fixed_freeze_self_forget_1000_forget_notop"

    # "BOUNDARYEXPAND_lr1e-05"
    # "BOUNDARYSHRINK_eps0.1_lr1e-05"

    # "DISTRIBUTIONAL_1000_standard_cov_diagonal_syntetic_fixed_embs_forget_notop"
    # "DISTRIBUTIONAL_1000_standard_cov_diagonal_syntetic_fixed_embs_self_forget_1000_forget_notop" # Should behave as freeze with no top due that embs are enable to be trained but the data lives on embs


    # "DISTRIBUTIONAL_1000_standard_cov_diagonal_syntetic_fixed_embs_forget_distances"
    # "DISTRIBUTIONAL_1000_standard_cov_diagonal_syntetic_fixed_embs_self_forget_1000_forget_distances"

    "SFU_damping0.0001_scale1.0_hessforget_freeze"
)


methods=(
    "EWC_10.0_lambda_MAS_embs_notop"
    "EWC_1.0_lambda_MAS_embs_notop"

    "EWC_10.0_lambda_MAS_embs_confidence_penalty"
    "EWC_1.0_lambda_MAS_embs_confidence_penalty"

)

datasets=("cifar10" "mnist")
experiments=("resnet18")
classes=(0 1 2 3 4 5 6 7 8 9) # CIFAR10 / MNIST classes
datasets=("cifar100")
classes=(4 1 70 10 51 39 5 24 3 76 60 15 34 26 11 29 80 96 8 89) # Classes selected of cifar100


method=${methods[$SLURM_ARRAY_TASK_ID]}

for experiment in "${experiments[@]}"; do
    datasets=("cifar10" "mnist")
    classes=(0 1 2 3 4 5 6 7 8 9) # CIFAR10 / MNIST classes
    for dataset in "${datasets[@]}"; do
        for class in "${classes[@]}"; do
            # for epoch2 in $(ls models/${experiment}_${dataset}_classes_to_remove_${class}/${method}/original_best/checkpoint_epoch_*.pt | sed -E 's/.*checkpoint_epoch_([0-9]+)\.pt/\1/'); do
            #     echo "Starting extraction process for ${experiment} epc ${epoch2} on ${dataset} original_best with method ${method}"
            #     python src/training/main_extractor.py -o results --model-directory models -e ${experiment}_${dataset} -m ${experiment} -d datasets --dataset ${dataset} --data-mode CR --classes-to-remove ${class} --unlearning-method ${method} --version original_best --checkpoint ${epoch2}
            # done

            echo "Starting extraction process for ${experiment} epc best on ${dataset} original_best with method ${method}"
            python src/training/main_extractor.py -o results --model-directory models -e ${experiment}_${dataset} -m ${experiment} -d datasets --dataset ${dataset} --data-mode CR --classes-to-remove ${class} --unlearning-method ${method} --version original_best --checkpoint best
            python src/training/main_extractor.py -o results --model-directory models -e ${experiment}_${dataset} -m ${experiment} -d datasets --dataset ${dataset} --data-mode CR --classes-to-remove ${class} --unlearning-method ${method} --version original_best --checkpoint 00
            # python src/training/main_extractor.py -o results --model-directory models -e ${experiment}_${dataset} -m ${experiment} -d datasets --dataset ${dataset} --data-mode CR --classes-to-remove ${class} --version original --checkpoint best
        done
    done

    datasets=("cifar100")
    classes=(4 1 70 10 51 39 5 24 3 76 60 15 34 26 11 29 80 96 8 89) # Classes selected of cifar100
    for dataset in "${datasets[@]}"; do
        for class in "${classes[@]}"; do
            # for epoch2 in $(ls models/${experiment}_${dataset}_classes_to_remove_${class}/${method}/original_best/checkpoint_epoch_*.pt | sed -E 's/.*checkpoint_epoch_([0-9]+)\.pt/\1/'); do
            #     echo "Starting extraction process for ${experiment} epc ${epoch2} on ${dataset} original_best with method ${method}"
            #     python src/training/main_extractor.py -o results --model-directory models -e ${experiment}_${dataset} -m ${experiment} -d datasets --dataset ${dataset} --data-mode CR --classes-to-remove ${class} --unlearning-method ${method} --version original_best --checkpoint ${epoch2}
            # done

            echo "Starting extraction process for ${experiment} epc best on ${dataset} original_best with method ${method}"
            python src/training/main_extractor.py -o results --model-directory models -e ${experiment}_${dataset} -m ${experiment} -d datasets --dataset ${dataset} --data-mode CR --classes-to-remove ${class} --unlearning-method ${method} --version original_best --checkpoint best
            python src/training/main_extractor.py -o results --model-directory models -e ${experiment}_${dataset} -m ${experiment} -d datasets --dataset ${dataset} --data-mode CR --classes-to-remove ${class} --unlearning-method ${method} --version original_best --checkpoint 00
            # python src/training/main_extractor.py -o results --model-directory models -e ${experiment}_${dataset} -m ${experiment} -d datasets --dataset ${dataset} --data-mode CR --classes-to-remove ${class} --version original --checkpoint best
        done
    done
done
conda deactivate
