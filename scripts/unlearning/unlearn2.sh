#!/bin/bash 
#SBATCH -c 16
#SBATCH --mem 30g
#SBATCH --job-name unswint
#SBATCH --output=slrums/output_%A_%a.out # STDOUT
#SBATCH --gres=gpu:1
#SBATCH --array=[0]%2

eval "$(conda shell.bash hook)"

conda activate unlearning
export PYTHONPATH='.'



experiments=("swin_t") #  "swin_t" "efficientnet_v2_s" "vit_b_16" "resnet18"

methods=(

    "--unlearning-method SSD --SSD-coefficients MAS --SSD-lambda 1.0 --SSD-alpha 10.0  --freeze embs"
    # "--unlearning-method NOISY --loss-forget notop --syntetic-samples 1000 --fixed-syntetic --covariance-type ledoit_wolf"
    "--unlearning-method NOISY --loss-forget notop --syntetic-samples 1000 --fixed-syntetic --emb-unfreeze embs --covariance-type ledoit_wolf"
    "--unlearning-method DISTRIBUTIONAL --loss-forget notop --syntetic-samples 1000 --fixed-syntetic --emb-unfreeze embs --covariance-type ledoit_wolf"

    "--unlearning-method DISTRIBUTIONAL --loss-forget notopalign --syntetic-samples 1000 --emb-unfreeze embs --fixed-syntetic --covariance-type ledoit_wolf"
    "--unlearning-method DISTRIBUTIONAL --loss-forget notopexpandalign --syntetic-samples 1000 --emb-unfreeze embs --fixed-syntetic --covariance-type ledoit_wolf"
    "--unlearning-method DISTRIBUTIONAL --loss-forget notopexpand --syntetic-samples 1000 --emb-unfreeze embs --fixed-syntetic --covariance-type ledoit_wolf"
    "--unlearning-method DISTRIBUTIONAL --loss-forget loglikelihood --syntetic-samples 1000 --emb-unfreeze embs --fixed-syntetic --covariance-type ledoit_wolf"

    "--unlearning-method DISTRIBUTIONAL --loss-forget distances --emb-unfreeze embs --syntetic-samples 1000 --fixed-syntetic --covariance-type ledoit_wolf"

    "--unlearning-method DISTRIBUTIONAL --loss-forget notop --emb-unfreeze embs --update-syntetic --syntetic-samples 1000 --fixed-syntetic --covariance-type ledoit_wolf"
    
    "--unlearning-method DISTRIBUTIONAL --loss-forget notopalign --syntetic-samples 1000 --emb-unfreeze embs --fixed-syntetic --update-syntetic --covariance-type ledoit_wolf"
    "--unlearning-method DISTRIBUTIONAL --loss-forget notopexpandalign --syntetic-samples 1000 --emb-unfreeze embs --fixed-syntetic --update-syntetic --covariance-type ledoit_wolf"
    "--unlearning-method DISTRIBUTIONAL --loss-forget notopexpand --syntetic-samples 1000 --emb-unfreeze embs --fixed-syntetic --update-syntetic --covariance-type ledoit_wolf"
    "--unlearning-method DISTRIBUTIONAL --loss-forget loglikelihood --syntetic-samples 1000 --emb-unfreeze embs --fixed-syntetic --update-syntetic --covariance-type ledoit_wolf"

    "--unlearning-method DISTRIBUTIONAL --loss-forget distances --emb-unfreeze embs --syntetic-samples 1000 --fixed-syntetic --update-syntetic --covariance-type ledoit_wolf"
    
    # "--unlearning-method DISTRIBUTIONAL --loss-forget confidence_penalty --syntetic-samples 1000 --fixed-syntetic --covariance-type diagonal"


    "--unlearning-method SSD --SSD-coefficients MAS --SSD-lambda 1.0 --SSD-alpha 5.0  --freeze embs"
    # "--unlearning-method NOISY --loss-forget confidence_penalty --syntetic-samples 1000 --fixed-syntetic --covariance-type diagonal --clustering"
    "--unlearning-method RANDOMLABEL --rl-retain-fraction 0.0  --freeze embs"     
    # "--unlearning-method DISTRIBUTIONAL --loss-forget confidence_penalty --syntetic-samples 1000 --fixed-syntetic --covariance-type diagonal --clustering"


    
    "--unlearning-method EWC --loss-forget notop --emb-unfreeze embs --lambda-ewc 1.0 --ewc-coefficients MAS"
    
    "--unlearning-method EWC --loss-forget notop --emb-unfreeze embs --lambda-ewc 10.0 --ewc-coefficients MAS"

    # "--unlearning-method RANDOMLABEL --rl-retain-fraction 0.1"

    "--unlearning-method BADTEACHER --bt-temperature 1.0 --no-bt-use-retain  --freeze embs"

    "--unlearning-method BOUNDARYEXPAND  --freeze embs"

    "--unlearning-method BOUNDARYSHRINK --freeze embs"

    "--unlearning-method ISPF --freeze embs" # Cancelled: it does not work well and takes too long

    "--unlearning-method FISHERFORGETTING --fisher-mode subtract  --freeze embs"

    "--unlearning-method SFU --sfu-hessian-data forget  --freeze embs"

    "--unlearning-method DELETE --freeze-delete embs"

    "--unlearning-method RANDOMLABEL --rl-retain-fraction 0.0  --freeze embs --use-forbidden-labels"

    "--unlearning-method RETRAIN"

    "--unlearning-method SSD --SSD-coefficients MAS --SSD-lambda 1.0 --SSD-alpha 1.0  --freeze embs"

    "--unlearning-method SSD --SSD-coefficients MAS --SSD-lambda 1.0 --SSD-alpha 20.0  --freeze embs"

    "--unlearning-method SSD --SSD-coefficients MAS --SSD-lambda 0.5 --SSD-alpha 1.0  --freeze embs"

    "--unlearning-method SSD --SSD-coefficients MAS --SSD-lambda 0.5 --SSD-alpha 5.0  --freeze embs"

    "--unlearning-method SSD --SSD-coefficients MAS --SSD-lambda 0.5 --SSD-alpha 10.0  --freeze embs"

    "--unlearning-method SSD --SSD-coefficients MAS --SSD-lambda 0.5 --SSD-alpha 20.0  --freeze embs"

    "--unlearning-method SSD --SSD-coefficients MAS --SSD-lambda 0.1 --SSD-alpha 1.0  --freeze embs"

    "--unlearning-method SSD --SSD-coefficients MAS --SSD-lambda 0.1 --SSD-alpha 5.0  --freeze embs"

    "--unlearning-method SSD --SSD-coefficients MAS --SSD-lambda 0.1 --SSD-alpha 10.0  --freeze embs"

    "--unlearning-method SSD --SSD-coefficients MAS --SSD-lambda 0.1 --SSD-alpha 20.0  --freeze embs"

    # "--unlearning-method RANDOMLABEL --rl-retain-fraction 0.0  --freeze embs --use-forbidden-labels"


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

    # "--unlearning-method DISTRIBUTIONAL --loss-forget notop --syntetic-samples 1000 --fixed-syntetic --emb-unfreeze embs --covariance-type ledoit_wolf"
    "--unlearning-method DISTRIBUTIONAL --loss-forget corrected_confidence_penalty --syntetic-samples 1000 --fixed-syntetic --emb-unfreeze embs --covariance-type ledoit_wolf"

)

methods=(
    "--unlearning-method NOISY --loss-forget confidence_penalty --syntetic-samples 1000 --fixed-syntetic --emb-unfreeze embs --covariance-type ledoit_wolf"
    "--unlearning-method NOISY --loss-forget notop --syntetic-samples 1000 --fixed-syntetic --emb-unfreeze embs --covariance-type ledoit_wolf"
)

methods=(

    "--unlearning-method DISTRIBUTIONAL --loss-forget notop --syntetic-samples 1000 --fixed-syntetic --emb-unfreeze embs --covariance-type ledoit_wolf"
    # "--unlearning-method DISTRIBUTIONAL --loss-forget corrected_confidence_penalty --syntetic-samples 1000 --fixed-syntetic --emb-unfreeze embs --covariance-type ledoit_wolf"

)

method=${methods[$SLURM_ARRAY_TASK_ID]}

for experiment in "${experiments[@]}"; do
    datasets=("cifar10" "mnist")
    classes=(0 1 2 3 4 5 6 7 8 9) # Classes selected of cifar100
    for dataset in "${datasets[@]}"; do
        for class in "${classes[@]}"; do

            echo "Starting unlearning process for ${experiment} of ${dataset} with method ${method}"
            python src/unlearning/main_unlearning.py -o models -e ${experiment}_${dataset} -m ${experiment} --save-epoch 5 --max-epoch 30 -d datasets --dataset ${dataset} --data-mode CR --train-batch-size 64 --val-batch-size 16 --seed 42 --classes-to-remove ${class} --version original --epoch best ${method}
    
        done
    done
    datasets=("cifar100")
    classes=(4 1 70 10 51 39 5 24 3 76 60 15 34 26 11 29 80 96 8 89) # Classes selected of cifar100
    for dataset in "${datasets[@]}"; do
        for class in "${classes[@]}"; do

            echo "Starting unlearning process for ${experiment} of ${dataset} with method ${method}"
            python src/unlearning/main_unlearning.py -o models -e ${experiment}_${dataset} -m ${experiment} --save-epoch 5 --max-epoch 30 -d datasets --dataset ${dataset} --data-mode CR --train-batch-size 64 --val-batch-size 16 --seed 42 --classes-to-remove ${class} --version original --epoch best ${method}
    
        done
    done
done

conda deactivate
