#!/bin/bash 
#SBATCH -c 2
#SBATCH --mem 5g
#SBATCH --job-name select_classes_cifar_100
#SBATCH --output=slrums/output_%A_%a.out # STDOUT

eval "$(conda shell.bash hook)"

conda activate unlearning
export PYTHONPATH='.'

seeds=(42 43 44 45 46 47 48 49 50 51)
classes_number_to_delete=(5 10 20 50)

for seed in "${seeds[@]}"; do
    # for num_classes in "${classes_number_to_delete[@]}"; do
    #     python src/training/select_classes_to_delete_cifar100.py --output-file "selected_classes_to_delete_seed_${seed}_num_${num_classes}.txt" --seed $seed --num-classes-to-delete $num_classes --use-all
    # done

    python src/training/select_random_numbers.py --output-file "selected_classes_to_delete_seed_${seed}_for_cifar10mnist.txt" --seed $seed --num-classes-to-delete 5 --num-classes 10
done
conda deactivate
