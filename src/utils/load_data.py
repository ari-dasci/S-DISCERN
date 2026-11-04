import os
import numpy as np
import torch
from PIL import Image
from torchvision import datasets as tv_datasets


# For backdoor attacks, we will add a colored pixel patch to the top-left corner of the image. Each backdoor attack will have a different color to make them easily distinguishable.
# Depending on the number of backdoor attacks we want to support, we can define a list of colors. For simplicity, we will use a predefined set of colors and cycle through them if there are more attacks than colors.
BACKDOOR_COLORS = [
    (0, 0, 0),        # Black
    (255, 255, 255),  # White
    (255, 0, 0),      # Red
    (0, 0, 255),      # Blue
    (0, 255, 0),      # Green
    (255, 255, 0),    # Yellow
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Cyan
    (128, 0, 0),      # Maroon
    (128, 128, 0),    # Olive
]

valid_datasets = ["CIFAR10", "CIFAR100", "MNIST", "LETTERS", "DTD"]
valid_datasets_sorted = sorted(valid_datasets, key=len, reverse=True)

DATASET_CLASSES = {
    "CIFAR10": 10,
    "CIFAR100": 100,
    "MNIST": 10,
    "DTD": 43,
    "LETTERS": 26
}

def extract_datasets_from_filename(filename):
    for ds1 in valid_datasets_sorted:
        if filename.startswith("./folds/" + ds1):
            return ds1
    return None  # No valid match found

def number_dataset_classes(dataset_name):
    """Return the number of classes for a given dataset name.
    
    Args:
        dataset_name: One of CIFAR10, CIFAR100, MNIST, DTD, LETTERS (case-insensitive)
    """
    dataset_name_upper = dataset_name.upper()
    if dataset_name_upper in DATASET_CLASSES:
        return DATASET_CLASSES[dataset_name_upper]
    else:
        raise ValueError(f"Dataset '{dataset_name}' not recognized. Valid options are: {list(DATASET_CLASSES.keys())}")

def add_pixel_backdoor(img, color, corner_size=4):
    """
    Add a colored pixel patch to the top-left corner of an image.
    
    Args:
        img: PIL Image or numpy array
        color: tuple (R, G, B)
        corner_size: size of the corner patch (default 4x4)
    
    Returns:
        PIL Image with backdoor pixel added
    """
    from PIL import Image
    import numpy as np
    
    # Convert to PIL if needed
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img.astype('uint8'))
    
    # Make a copy to avoid modifying original
    img_with_backdoor = img.copy()
    
    # Add colored patch to top-left corner
    for i in range(corner_size):
        for j in range(corner_size):
            img_with_backdoor.putpixel((i, j), color)
    
    return img_with_backdoor


def _load_raw_dataset(dataset_name, data_root, train=True):
    """
    Load a dataset split from torchvision.
    Returns (images, labels) as lists of RGB PIL Images and int labels.

    Args:
        dataset_name : one of CIFAR10, CIFAR100, MNIST, DTD, LETTERS
        data_root    : root directory where torchvision stores the data
        train        : True → training split, False → test split

    Supported datasets: CIFAR10, CIFAR100, MNIST, DTD, LETTERS (EMNIST).
    """


    dataset_name_upper = dataset_name.upper()

    def _extract(ds):
        """Extract (PIL Image, int label) from a torchvision dataset."""
        imgs, lbls = [], []
        for img, label in ds:
            if not isinstance(img, Image.Image):
                img = Image.fromarray(np.uint8(img))
            imgs.append(img)
            lbls.append(int(label))
        return imgs, lbls

    if dataset_name_upper == "CIFAR10":
        ds = tv_datasets.CIFAR10(root=data_root, train=train, download=False)
        images, labels = _extract(ds)

    elif dataset_name_upper == "CIFAR100":
        ds = tv_datasets.CIFAR100(root=data_root, train=train, download=False)
        images, labels = _extract(ds)

    elif dataset_name_upper == "MNIST":
        # EMNIST already contains the MNIST split, reuse it to avoid a separate download.
        ds = tv_datasets.EMNIST(root=data_root, split="mnist", train=train, download=False)
        images, labels = _extract(ds)

    elif dataset_name_upper == "DTD":
        # DTD uses named splits instead of train=True/False
        if train:
            ds_train = tv_datasets.DTD(root=data_root, split="train", download=False)
            ds_val   = tv_datasets.DTD(root=data_root, split="val",   download=False)
            imgs_t, lbls_t = _extract(ds_train)
            imgs_v, lbls_v = _extract(ds_val)
            images = imgs_t + imgs_v
            labels = lbls_t + lbls_v
        else:
            ds = tv_datasets.DTD(root=data_root, split="test", download=False)
            images, labels = _extract(ds)

    elif dataset_name_upper == "LETTERS":
        ds = tv_datasets.EMNIST(root=data_root, split="letters", train=train, download=True)
        images, labels = _extract(ds)
        # EMNIST-letters labels are 1-based (A=1 … Z=26) → remap to 0-25
        labels = [l - 1 for l in labels]

    else:
        raise ValueError(
            f"Dataset '{dataset_name}' not supported. "
            f"Valid options: {valid_datasets}"
        )

    return images, labels


def load_data_backdoor(args, train=True):
    """Loads the backdoored data according to the given arguments"""
    dataset_name = args.dataset
    data_root = args.data_path
    dataset_backdoors = args.backdoors
    percentages_backdoors = args.percentages_backdoors
    objectives_backdoors = args.objectives_backdoors

    # Load dataset directly from torchvision
    X_train, y_train = _load_raw_dataset(dataset_name, data_root, train=train)

    x_train_unlearn = []
    y_train_unlearn = []

    all_poisoned_indices = set()

    for i in range(len(dataset_backdoors)):
        np.random.seed(2809 + i) # Set seed for reproducibility, feel free to change it!!!
        attacked_class = dataset_backdoors[i]
        percentage_backdoor = percentages_backdoors[i]
        objective_backdoor = objectives_backdoors[i]
        
        # Select color for this backdoor attack
        backdoor_color = BACKDOOR_COLORS[i % len(BACKDOOR_COLORS)]
        
        print(f"\nBackdoor {i}: Attack class {attacked_class} → Target class {objective_backdoor} with color {backdoor_color}")
        
        # Find indices of samples from the attack class
        attacked_class_indices = [idx for idx, label in enumerate(y_train) if label == attacked_class]
        
        if not attacked_class_indices:
            print(f"Warning: Class {attacked_class} not found in training data!")
            continue
        
        # Select random percentage of attacked class samples
        n_backdoor_samples = int(len(attacked_class_indices) * percentage_backdoor)
        if n_backdoor_samples == 0:
            print(f"Warning: Percentage {percentage_backdoor*100}% of class {attacked_class} results in 0 samples!")
            continue
        
        # Randomly select indices to poison
        rng = np.random.default_rng(2809 + i)
        selected_indices = rng.choice(attacked_class_indices, size=n_backdoor_samples, replace=False)
        
        # Add backdoor to selected samples
        all_poisoned_indices.update(selected_indices)
        for idx in selected_indices:
            img = X_train[idx]
            
            # Add pixel backdoor
            img_poisoned = add_pixel_backdoor(img, backdoor_color, corner_size=4)
            
            x_train_unlearn.append(img_poisoned)
            y_train_unlearn.append(objective_backdoor)

            X_train[idx] = img_poisoned
            # Change label to objective class
            y_train[idx] = objective_backdoor

    # Build retain set excluding ALL poisoned samples across all attacks
    X_retain = [img for j, img in enumerate(X_train) if j not in all_poisoned_indices]
    y_retain = [label for j, label in enumerate(y_train) if j not in all_poisoned_indices]
        
    forbidden_labels = [] # Labels that should not be used for syntetic samples, in this case none because we are not removing any class

    return X_train, y_train, X_retain, y_retain, x_train_unlearn, y_train_unlearn, forbidden_labels

def load_data_class(args, objective=[9], train=True):
    dataset_name = args.extract_dataset
    data_root = args.data_path

    # Load dataset directly from torchvision
    X_train, y_train = _load_raw_dataset(dataset_name, data_root, train=train)

    x_train_unlearn = [img for img, label in zip(X_train, y_train) if label in objective]
    y_train_unlearn = [label for label in y_train if label in objective]

    X_retain = [img for img, label in zip(X_train, y_train) if label not in objective]
    y_retain = [label for label in y_train if label not in objective]
    
    forbidden_labels = objective # Assuming last label is the outlier class

    return X_train, y_train, X_retain, y_retain, x_train_unlearn, y_train_unlearn, forbidden_labels

def load_data_general(args, train=True):
    if args.data_mode == "backdoor":
        return load_data_backdoor(args, train=train)
    elif args.data_mode == "CR" or args.data_mode == "train":
        if args.extract_dataset != args.dataset:
            return load_data_class(args, train=train, objective=[])
        else:
            return load_data_class(args, train=train, objective=args.classes_to_remove if args.classes_to_remove is not None else [])
    else:
        raise ValueError("Invalid data mode specified")