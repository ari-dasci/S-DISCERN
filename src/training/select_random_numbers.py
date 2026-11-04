### This python script is used to select the classes to delete for CIFAR100 dataset.
### It will select randomly one class from each superclass to delete and save the selected classes in a txt file.

import random
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Select classes to delete for CIFAR100 dataset.')
    parser.add_argument('--output-file', type=str, default='selected_classes_to_delete.txt', help='Output file to save the selected classes to delete.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--num-classes-to-delete', type=int, default=5, help='Number of classes to delete (default: 5).')
    parser.add_argument('--num-classes', type=int, default=10, help='Total number of classes in the dataset (default: 10).')
    args = parser.parse_args()

    random.seed(args.seed)
    classes = list(range(args.num_classes))
    selected_classes_to_delete = []

        
    selected_classes_to_delete = random.sample(classes, args.num_classes_to_delete)


    with open(args.output_file, 'w') as f:
        for cls in selected_classes_to_delete:
            f.write(str(cls) + '\n')

    print(f'Selected classes to delete: {selected_classes_to_delete}')
    print(f'Selected classes saved to {args.output_file}')