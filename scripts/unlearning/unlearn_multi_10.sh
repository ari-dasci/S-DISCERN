#!/bin/bash 
#SBATCH -c 16
#SBATCH --mem 30g
#SBATCH --job-name multi10
#SBATCH --output=slrums/output_%A_%a.out # STDOUT
#SBATCH --gres=gpu:1
#SBATCH --array=12%3

eval "$(conda shell.bash hook)"

conda activate unlearning
export PYTHONPATH='.'


methods=(

    "--unlearning-method RETRAIN"

    "--unlearning-method SSD --SSD-coefficients MAS --SSD-lambda 1.0 --SSD-alpha 10.0  --freeze embs"
    "--unlearning-method NOISY --loss-forget notop --syntetic-samples 1000 --fixed-syntetic --covariance-type ledoit_wolf"
    "--unlearning-method NOISY --loss-forget notop --syntetic-samples 1000 --fixed-syntetic --emb-unfreeze embs --covariance-type ledoit_wolf"
    # "--unlearning-method DISTRIBUTIONAL --loss-forget confidence_penalty --syntetic-samples 1000 --fixed-syntetic --covariance-type diagonal"


    "--unlearning-method SSD --SSD-coefficients MAS --SSD-lambda 1.0 --SSD-alpha 5.0  --freeze embs"
    # "--unlearning-method NOISY --loss-forget confidence_penalty --syntetic-samples 1000 --fixed-syntetic --covariance-type diagonal --clustering"
    "--unlearning-method RANDOMLABEL --rl-retain-fraction 0.0  --freeze embs"
    # "--unlearning-method DISTRIBUTIONAL --loss-forget confidence_penalty --syntetic-samples 1000 --fixed-syntetic --covariance-type diagonal --clustering"


    
    # "--unlearning-method EWC --loss-forget notop --lambda-ewc 1.0 --ewc-coefficients MAS"
    
    # "--unlearning-method EWC --loss-forget notop --lambda-ewc 10.0 --ewc-coefficients MAS"

    # "--unlearning-method RANDOMLABEL --rl-retain-fraction 0.1"

    "--unlearning-method BADTEACHER --bt-temperature 1.0 --no-bt-use-retain  --freeze embs"

    "--unlearning-method BOUNDARYEXPAND  --freeze embs"

    "--unlearning-method BOUNDARYSHRINK --freeze embs"

    "--unlearning-method ISPF --freeze embs"

    "--unlearning-method FISHERFORGETTING --fisher-mode subtract  --freeze embs"

    "--unlearning-method SFU --sfu-hessian-data forget  --freeze embs"

    "--unlearning-method DELETE --freeze-delete embs"

    "--unlearning-method RANDOMLABEL --rl-retain-fraction 0.0  --freeze embs --use-forbidden-labels"

    "--unlearning-method RANDOMLABEL --rl-retain-fraction 0.0  --freeze freeze --use-forbidden-labels"

    # "--unlearning-method DISTRIBUTIONAL --loss-forget notop --syntetic-samples 1000 --fixed-syntetic --covariance-type diagonal --clustering"
    # "--unlearning-method DISTRIBUTIONAL --loss-forget notop --syntetic-samples 1000 --fixed-syntetic --covariance-type diagonal"

    # "--unlearning-method DISTRIBUTIONAL --loss-forget notop --syntetic-samples 1000 --fixed-syntetic --covariance-type diagonal --clustering --self-forget --forget-syntetic-samples 1000"
    # "--unlearning-method DISTRIBUTIONAL --loss-forget notop --syntetic-samples 1000 --fixed-syntetic --covariance-type diagonal --self-forget --forget-syntetic-samples 1000"

    # "--unlearning-method DISTRIBUTIONAL --loss-forget notop --emb-unfreeze --update-syntetic --syntetic-samples 1000 --fixed-syntetic --covariance-type diagonal"

    # "--unlearning-method DISTRIBUTIONAL --loss-forget notop --emb-unfreeze --update-syntetic --syntetic-samples 1000 --fixed-syntetic --covariance-type diagonal --self-forget --forget-syntetic-samples 1000"

    # "--unlearning-method DISTRIBUTIONAL --loss-forget distances --emb-unfreeze --update-syntetic --syntetic-samples 1000 --fixed-syntetic --covariance-type diagonal"

    # "--unlearning-method DISTRIBUTIONAL --loss-forget distances --emb-unfreeze --update-syntetic --syntetic-samples 1000 --fixed-syntetic --covariance-type diagonal --self-forget --forget-syntetic-samples 1000"
)

methods=(
    "--unlearning-method DISTRIBUTIONAL --loss-forget notop --syntetic-samples 1000 --fixed-syntetic --emb-unfreeze embs --covariance-type ledoit_wolf"
    "--unlearning-method DISTRIBUTIONAL --loss-forget notop --syntetic-samples 1000 --fixed-syntetic --update-syntetic --emb-unfreeze embs --covariance-type ledoit_wolf"
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

    "--unlearning-method RETRAIN"

    "--unlearning-method DISTRIBUTIONAL --loss-forget corrected_confidence_penalty --syntetic-samples 1000 --fixed-syntetic --emb-unfreeze embs --covariance-type ledoit_wolf"
    "--unlearning-method DISTRIBUTIONAL --loss-forget notop --syntetic-samples 1000 --fixed-syntetic --emb-unfreeze embs --covariance-type ledoit_wolf"
    
    "--unlearning-method NOISY --loss-forget confidence_penalty --syntetic-samples 1000 --fixed-syntetic --emb-unfreeze embs --covariance-type ledoit_wolf"
    "--unlearning-method NOISY --loss-forget notop --syntetic-samples 1000 --fixed-syntetic --emb-unfreeze embs --covariance-type ledoit_wolf"

    "--unlearning-method DELETE --freeze-delete embs"

    "--unlearning-method RANDOMLABEL --rl-retain-fraction 0.0  --freeze embs --use-forbidden-labels"

    "--unlearning-method EWC --loss-forget notop --lambda-ewc 1.0 --ewc-coefficients MAS --emb-unfreeze embs"
    "--unlearning-method EWC --loss-forget notop --lambda-ewc 10.0 --ewc-coefficients MAS --emb-unfreeze embs"

    "--unlearning-method EWC --loss-forget confidence_penalty --lambda-ewc 1.0 --ewc-coefficients MAS --emb-unfreeze embs"
    "--unlearning-method EWC --loss-forget confidence_penalty --lambda-ewc 10.0 --ewc-coefficients MAS --emb-unfreeze embs"

    "--unlearning-method BADTEACHER --bt-temperature 1.0 --no-bt-use-retain  --freeze embs"

    "--unlearning-method BOUNDARYEXPAND  --freeze embs"

    "--unlearning-method BOUNDARYSHRINK --freeze embs"

    "--unlearning-method SSD --SSD-coefficients MAS --SSD-lambda 0.1 --SSD-alpha 1.0  --freeze embs"
    "--unlearning-method SSD --SSD-coefficients MAS --SSD-lambda 1.0 --SSD-alpha 10.0  --freeze embs"
    "--unlearning-method SSD --SSD-coefficients MAS --SSD-lambda 1.0 --SSD-alpha 5.0  --freeze embs"

)

method=${methods[$SLURM_ARRAY_TASK_ID]}

seeds=(42 43 44 45 46 47 48 49 50 51)

seed_42="81 14 3 94 35 31 28 17 13 86"

seed_43="4 36 89 97 18 59 47 85 12 58"

seed_44="52 66 69 89 14 22 48 28 37 3"

seed_45="34 53  62 32 10 38 43 2 9 61"
seed_46="9 51 5 75 29 81 66 19 74 68"
seed_47="45 8 55 70 58 73 43 32 65 49"

seed_48="70 40 16 71 91 68 38 64 24 97"
seed_49="8 44 52 14 41 96 70 65 6 87"
seed_50="63 34 46 81 31 88 60 98 42 10"
seed_51="31 64 70 20 29 32 50 59 93 89"

experiments=("swin_t" "efficientnet_v2_s" "resnet18") # "swin_t" "efficientnet_v2_s" "vit_b_16" "resnet18"
datasets=("cifar100")

for experiment in "${experiments[@]}"; do
    for dataset in "${datasets[@]}"; do
        for seed in "${seeds[@]}"; do

            classes_to_remove_var="seed_${seed}[@]"
            classes_to_remove=("${!classes_to_remove_var}")
            echo "$classes_to_remove"
            echo "Starting unlearning process for ${experiment} of ${dataset} with method ${method}"
            python src/unlearning/main_unlearning.py -o models -e ${experiment}_${dataset} -m ${experiment} --save-epoch 5 --max-epoch 30 -d datasets --dataset ${dataset} --data-mode CR --train-batch-size 64 --val-batch-size 16 --seed $seed --classes-to-remove $classes_to_remove --version original --epoch best ${method}
        done
    done
done

conda deactivate
