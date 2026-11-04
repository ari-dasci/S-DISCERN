#!/bin/bash 
#SBATCH -c 16
#SBATCH --mem 30g
#SBATCH --job-name unlearn
#SBATCH --output=slrums/output_%A_%a.out # STDOUT
#SBATCH --gres=gpu:1
#SBATCH --array=0-19%3

eval "$(conda shell.bash hook)"

conda activate unlearning
export PYTHONPATH='.'


experiments=("vit_b_16" "resnet18")

datasets=("cifar100")
classes=(4 1 70 10 51 39 5 24 3 76 60 15 34 26 11 29 80 96 8 89) # Classes selected of cifar100

experiment=${experiments[$SLURM_ARRAY_TASK_ID]}
dataset=${datasets[$SLURM_ARRAY_TASK_ID]}


methods=(

    "--unlearning-method RETRAIN"
)

dataset="cifar100"
method="--unlearning-method RETRAIN"
class=${classes[$SLURM_ARRAY_TASK_ID]}


for experiment in "${experiments[@]}"; do
    echo "Starting unlearning process for ${experiment} of ${dataset} with method ${method}"
    python src/unlearning/main_unlearning.py -o models -e ${experiment}_${dataset} -m ${experiment} --save-epoch 5 --max-epoch 30 -d datasets --dataset ${dataset} --data-mode CR --train-batch-size 64 --val-batch-size 16 --seed 42 --classes-to-remove ${class} --version original --epoch best ${method}
done

conda deactivate
