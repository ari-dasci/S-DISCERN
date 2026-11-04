#!/bin/bash 
#SBATCH -c 16
#SBATCH --mem 30g
#SBATCH --job-name train
#SBATCH --output=slrums/output_%A_%a.out # STDOUT
#SBATCH --gres=gpu:1
#SBATCH --array=0%3

eval "$(conda shell.bash hook)"

conda activate unlearning
export PYTHONPATH='.'


experiments=("swin_t") # "efficientnet_v2_s"
datasets=("cifar10" "cifar100" "mnist")

for experiment in "${experiments[@]}"; do
    for dataset in "${datasets[@]}"; do

            echo "Starting training process for ${experiment} of ${dataset} with method original"
            python src/training/main_training.py -o models -e ${experiment}_${dataset} -m ${experiment} --save-epoch 5 --max-epoch 30 -d datasets --dataset ${dataset} --data-mode CR --train-batch-size 64 --val-batch-size 16 --seed 42 --version original
            
    done
done
conda deactivate
