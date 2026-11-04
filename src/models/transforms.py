import sys
from torchvision import transforms

def load_transforms(args):
    # Define base transformation components based on model
    if args.base_model == "resnet18":
        transform_components = [transforms.ToTensor(), transforms.Resize(256), transforms.CenterCrop(224), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])]
    elif args.base_model == "efficientnet_v2_s":
        transform_components = [transforms.ToTensor(), transforms.Resize((384, 384)), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
    elif args.base_model == "swin_t":
        transform_components = [transforms.ToTensor(), transforms.Resize(232), transforms.CenterCrop(224), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    elif args.base_model == "vit_b_16":
        transform_components = [transforms.ToTensor(), transforms.Resize((224, 224)), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    else:
        print("ERROR UNRECOGNISED MODEL")
        sys.exit(-1)
    
    dataset_name = args.dataset.upper()
    if dataset_name in ["MNIST", "LETTERS"]:
        transform_components = [transforms.Grayscale(num_output_channels=3)] + transform_components
    
    # Create 3 separate instances
    transform_train = transforms.Compose(transform_components)
    transform_retain = transforms.Compose(transform_components)
    transform_forget = transforms.Compose(transform_components)

    return transform_train, transform_retain, transform_forget