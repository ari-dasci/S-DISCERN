from torchvision.datasets import CIFAR10, CIFAR100, EMNIST, DTD

def main():
    # Download CIFAR-10 dataset
    CIFAR10(root='./datasets', train=True, download=True)
    CIFAR10(root='./datasets', train=False, download=True)
    
    # Download CIFAR-100 dataset
    CIFAR100(root='./datasets', train=True, download=True)
    CIFAR100(root='./datasets', train=False, download=True)
    
    ## There can be an error wen downloading EMNIST, be carefull

    # # Download EMNIST-leters
    # EMNIST(root='./datasets', split='letters', train=True, download=True)
    # EMNIST(root='./datasets', split='letters', train=False, download=True)

    # # Download EMNIST-digits
    # EMNIST(root='./datasets', split='mnist', train=True, download=True)
    # EMNIST(root='./datasets', split='mnist', train=False, download=True)

    # Download DTD (textures)
    DTD(root='./datasets', split='train', download=True)
    DTD(root='./datasets', split='val', download=True)
    DTD(root='./datasets', split='test', download=True)

if __name__ == '__main__':
    main()