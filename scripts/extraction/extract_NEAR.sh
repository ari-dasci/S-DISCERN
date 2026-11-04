#!/bin/bash 
#SBATCH -c 16
#SBATCH --mem 20g
#SBATCH --job-name exNEAR
#SBATC#H -o ./slrums/%j.out # STDOUT
#SBATCH --output=slrums/output_%A_%a.out # STDOUT
#SBATCH --gres=gpu:1
#SBATCH --array=[0-37]%2

eval "$(conda shell.bash hook)"

conda activate unlearning
export PYTHONPATH='.'
methods=(
    "SSD_MAS_lambda_1.0_alpha_10.0_embs"
    "NOISY_1000_standard_clustering_cov_ledoit_wolf_syntetic_fixed_embs_forget_notop"

    "DISTRIBUTIONAL_1000_standard_cov_ledoit_wolf_syntetic_fixed_embs_forget_notop"
    "DISTRIBUTIONAL_1000_standard_cov_ledoit_wolf_syntetic_fixed_embs_forget_notopalign"
    "DISTRIBUTIONAL_1000_standard_cov_ledoit_wolf_syntetic_fixed_embs_forget_notopexpand"
    "DISTRIBUTIONAL_1000_standard_cov_ledoit_wolf_syntetic_fixed_embs_forget_notopexpandalign"
    "DISTRIBUTIONAL_1000_standard_cov_ledoit_wolf_syntetic_fixed_embs_forget_loglikelihood"
    "DISTRIBUTIONAL_1000_standard_cov_ledoit_wolf_syntetic_fixed_embs_forget_distances"

    "DISTRIBUTIONAL_1000_standard_cov_ledoit_wolf_syntetic_moving_embs_forget_notop"
    "DISTRIBUTIONAL_1000_standard_cov_ledoit_wolf_syntetic_moving_embs_forget_notopalign"
    "DISTRIBUTIONAL_1000_standard_cov_ledoit_wolf_syntetic_moving_embs_forget_notopexpand"
    "DISTRIBUTIONAL_1000_standard_cov_ledoit_wolf_syntetic_moving_embs_forget_notopexpandalign"
    "DISTRIBUTIONAL_1000_standard_cov_ledoit_wolf_syntetic_moving_embs_forget_loglikelihood"
    "DISTRIBUTIONAL_1000_standard_cov_ledoit_wolf_syntetic_moving_embs_forget_distances"

    "SSD_MAS_lambda_1.0_alpha_5.0_embs"
    "RANDOMLABEL_noret_embs"
    "EWC_10.0_lambda_MAS_embsunfreeze_notop"
    "EWC_1.0_lambda_MAS_embsunfreeze_notop"
    "BADTEACHER_T1.0_forgetOnly_embs"
    "BOUNDARYEXPAND_lr1e-05_embs"
    
    "BOUNDARYSHRINK_eps0.1_lr1e-05_embs"   
    "ISPF_nz256_T1.0_embs"
    "FISHERFORGETTING_noise1.0_alpha1e-06_modesubtract_embs"
    "SFU_damping0.0001_scale1.0_hessforget_embs"
    "DELETE_embs"
    "RANDOMLABEL_noret_embs_forbidden"

    "RETRAIN"

    "SSD_MAS_lambda_1.0_alpha_20.0_embs"
    "SSD_MAS_lambda_1.0_alpha_1.0_embs"

    "SSD_MAS_lambda_0.5_alpha_1.0_embs"
    "SSD_MAS_lambda_0.5_alpha_5.0_embs"
    "SSD_MAS_lambda_0.5_alpha_10.0_embs"
    "SSD_MAS_lambda_0.5_alpha_20.0_embs"

    "SSD_MAS_lambda_0.1_alpha_1.0_embs"
    "SSD_MAS_lambda_0.1_alpha_5.0_embs"
    "SSD_MAS_lambda_0.1_alpha_10.0_embs"
    "SSD_MAS_lambda_0.1_alpha_20.0_embs"

)
datasets=("cifar10" "mnist") # efficientnet_v2_s
experiments=("swin_t")
classes=(0 1 2 3 4 5 6 7 8 9) # CIFAR10 / MNIST classes
datasets=("cifar100")
classes=(4 1 70 10 51 39 5 24 3 76 60 15 34 26 11 29 80 96 8 89) # Classes selected of cifar100


method=${methods[$SLURM_ARRAY_TASK_ID]}

for experiment in "${experiments[@]}"; do
    datasets=("cifar10")
    classes=(0 1 2 3 4 5 6 7 8 9) # CIFAR10 / MNIST classes
    for dataset in "${datasets[@]}"; do
        for class in "${classes[@]}"; do

            echo "Starting extraction process for ${experiment} epc best on ${dataset} original_best with method ${method}"
            python src/training/main_extractor.py -o results_NEAR --model-directory models -e ${experiment}_${dataset} -m ${experiment} -d datasets --dataset ${dataset} --extract-dataset cifar100 --data-mode CR --classes-to-remove ${class} --version original_best --checkpoint 00 --unlearning-method ${method}

            python src/training/main_extractor.py -o results_NEAR --model-directory models -e ${experiment}_${dataset} -m ${experiment} -d datasets --dataset ${dataset} --extract-dataset cifar100 --data-mode CR --classes-to-remove ${class} --version original_best --checkpoint best --unlearning-method ${method}

            # python src/training/main_extractor.py -o results_NEAR --model-directory models -e ${experiment}_${dataset} -m ${experiment} -d datasets --dataset ${dataset} --extract-dataset cifar100 --data-mode CR --classes-to-remove ${class} --version original --checkpoint best
            
            
        done
    done

    datasets=("cifar100")
    classes=(4 1 70 10 51 39 5 24 3 76 60 15 34 26 11 29 80 96 8 89) # Classes selected of cifar100
    for dataset in "${datasets[@]}"; do
        for class in "${classes[@]}"; do
            echo "Starting extraction process for ${experiment} epc best on ${dataset} original_best with method ${method}"
            python src/training/main_extractor.py -o results_NEAR --model-directory models -e ${experiment}_${dataset} -m ${experiment} -d datasets --dataset ${dataset} --extract-dataset cifar10 --data-mode CR --classes-to-remove ${class} --version original_best --checkpoint 00 --unlearning-method ${method}

            python src/training/main_extractor.py -o results_NEAR --model-directory models -e ${experiment}_${dataset} -m ${experiment} -d datasets --dataset ${dataset} --extract-dataset cifar10 --data-mode CR --classes-to-remove ${class} --version original_best --checkpoint best --unlearning-method ${method}

            # python src/training/main_extractor.py -o results_NEAR --model-directory models -e ${experiment}_${dataset} -m ${experiment} -d datasets --dataset ${dataset} --extract-dataset cifar10 --data-mode CR --classes-to-remove ${class} --version original --checkpoint best
            
        done
    done

    datasets=("mnist")
    classes=(0 1 2 3 4 5 6 7 8 9) # MNIST classes
    for dataset in "${datasets[@]}"; do
        for class in "${classes[@]}"; do
            echo "Starting extraction process for ${experiment} epc best on ${dataset} original_best with method ${method}"
            python src/training/main_extractor.py -o results_NEAR --model-directory models -e ${experiment}_${dataset} -m ${experiment} -d datasets --dataset ${dataset} --extract-dataset letters --data-mode CR --classes-to-remove ${class} --version original_best --checkpoint 00 --unlearning-method ${method}

            python src/training/main_extractor.py -o results_NEAR --model-directory models -e ${experiment}_${dataset} -m ${experiment} -d datasets --dataset ${dataset} --extract-dataset letters --data-mode CR --classes-to-remove ${class} --version original_best --checkpoint best --unlearning-method ${method}

            # python src/training/main_extractor.py -o results_NEAR --model-directory models -e ${experiment}_${dataset} -m ${experiment} -d datasets --dataset ${dataset} --extract-dataset letters --data-mode CR --classes-to-remove ${class} --version original --checkpoint best
        done
    done
done
conda deactivate
