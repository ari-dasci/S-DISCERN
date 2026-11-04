#!/bin/bash 
#SBATCH -c 16
#SBATCH --mem 30g
#SBATCH --job-name multi50
#SBATCH --output=slrums/output_%A_%a.out # STDOUT
#SBATCH --gres=gpu:1
#SBATCH --array=11%3

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

seed_42="81 14 3 94 35 31 28 17 13 86 69 11 75 54 4 97 88 27 29 64 77 84 71 25 89 53 93 57 95 0 20 90 43 79 19 82 67 6 5 24 62 22 68 58 38 16 51 2 46 99"
seed_43="4 36 89 18 59 47 85 97 12 58 76 63 77 2 65 55 73 94 79 70 84 50 22 7 13 15 48 19 91 20 93 88 16 49 64 34 23 24 99 3 37 60 11 90 53 10 96 17 61 33"
seed_44="52 66 69 89 14 22 48 28 37 3 92 95 72 1 12 20 65 38 81 93 75 51 41 45 9 43 36  84 88 83 8 85 34 40 13 91 4 94 24 6 30 59 68 57 97 16 47 15 49 90"
seed_45="34 53 62 32 10 38 43 2 9 61 1 14 36 15 39 83 4 79 8 17 99 19 27 54 91 52 18 7 84 22 41 96 45 40 12 74 11 89 48 21 20 50 24 44 47 33 42 82 66 46"
seed_46="9 51 5 75 96 29 81 66 19 74 68 4 3 8 40 80 6 87 17 70 37 67 42 11 50 61 46 91 60 59 99  58 14 55 28 43 56 49 45 93 69 83 38 22 88 57 21 94 25 76"
seed_47="45 8 55 70 58 73 43 32 65 49 50 78 5 53 84 3 13 0 33 61 69 1 40 72 54 29 30 66 91 99 56 94 4 47 96 93 25 15 89 86 46 95 2 98 21 6 23 37 24 92"
seed_48="70 40 16 71 91 68 38 64 24 55 21 19 84 13 63 83 28 27 65 96 81 88 60 82 3 9 54 8 97 85 20 80 7 37 43 79 12 53 15 17 94 41 0 56 36 18 98 72 22 58"
seed_49="8 44 52 14 41 70 65 6 87 5 35 19 58 90 80 69 64 28 77 36 73 47 51 3 31 17 12 54 48 95 45 34 81 92 97 26 99 85 16 68 38 60 20 63 75 96 57 61 37 27"
seed_50="63 34 46 81 31 88 60 42 10 68 40 28 86 71 91 19 44 12 83 89 94 24 8 92 54 11 77 76 55 0 29 43 98 87 13 25 56 17 57 96 37 69 73 62 39 65 67 59 7 26"
seed_51="31 64 70 20 99 29 97 32 50 93 59 90 67 80 43 75 46 49 35 51 0 14 3 68 33 4 61 22 44 38 47 57 24 77 56 21 60 79 58 66 6 52 91 73 69 65 76 37 89 11"

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
