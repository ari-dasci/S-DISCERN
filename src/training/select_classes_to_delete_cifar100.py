### This python script is used to select the classes to delete for CIFAR100 dataset.
### It will select randomly one class from each superclass to delete and save the selected classes in a txt file.

import random
import os
import argparse

CIFAR100_SUPERCLASES_AND_CLASSES = {
    'aquatic_mammals': ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
    'fish': ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
    'flowers': ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
    'food_containers': ['bottle', 'bowl', 'can', 'cup', 'plate'],
    'fruit_and_vegetables': ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
    'household_devices': ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
    'household_furniture': ['bed', 'chair', 'couch', 'table', 'wardrobe'],
    'insects': ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
    'large_carnivores': ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
    'outdoor_things': ['bridge', 'castle', 'house', 'road', 'skyscraper'],
    'outdoor_scenes': ['cloud', 'forest', 'mountain', 'plain', 'sea'],
    'Large_herbivores': ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
    'medium_mammals': ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
    'invertebrates': ['crab', 'lobster', 'snail', 'spider', 'worm'],
    'people': ['baby', 'boy', 'girl', 'man', 'woman'],
    'reptiles': ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
    'small_mammals': ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
    'trees': ['maple', 'oak', 'palm', 'pine', 'willow'],
    'vehicles_1': ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
    'vehicles_2': ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']
}

CIFAR100_CLASSES_NAME=[]

for superclass, classes in CIFAR100_SUPERCLASES_AND_CLASSES.items():
    CIFAR100_CLASSES_NAME.extend(classes)

CIFAR100_CLASSES_NAME = sorted(CIFAR100_CLASSES_NAME)

CIFAR100_CLASSES_NAME_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(CIFAR100_CLASSES_NAME)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Select classes to delete for CIFAR100 dataset.')
    parser.add_argument('--output-file', type=str, default='selected_classes_to_delete.txt', help='Output file to save the selected classes to delete.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--num-classes-to-delete', type=int, default=20, help='Number of classes to delete (default: 20, which is one class from each superclass).')
    parser.add_argument('--use-all', action='store_true', help='Instead of selecting one class from each superclass, select classes randomly from all classes.')
    args = parser.parse_args()

    random.seed(args.seed)
    selected_classes_to_delete = []

    if args.use_all:
        all_classes = CIFAR100_CLASSES_NAME
        
        selected_classes_to_delete = random.sample(all_classes, args.num_classes_to_delete)
    
    else:
        for superclass, classes in CIFAR100_SUPERCLASES_AND_CLASSES.items():
            selected_class = random.choice(classes)
            selected_classes_to_delete.append(selected_class)

    id_clases_to_delete = [CIFAR100_CLASSES_NAME_TO_IDX[cls] for cls in selected_classes_to_delete]

    with open(args.output_file, 'w') as f:
        for cls in id_clases_to_delete:
            f.write(str(cls) + '\n')

    print(f'Selected classes to delete: {selected_classes_to_delete}')
    print(f'Selected classes to delete: {id_clases_to_delete}')
    print(f'Selected classes saved to {args.output_file}')