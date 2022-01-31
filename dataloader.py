import torch 
from torch.utils.data import DataLoader
from torchvision import datasets, transforms 

def get_cifar_10(batch_size):
    transform = transforms.Compose([transforms.Resize(32), #3*32*32
                                    transforms.ToTensor()
                        ])

    data = datasets.CIFAR10(root="./", train=True, download=True, transform=transform)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
    return dataloader

    