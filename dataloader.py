import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
import os  
ROOT = os.environ.get("data_dir", "./data")
BATCH_SIZE = 64 
class InlierDataset(Dataset):
    def __init__(self,root, inlier, transform=None):
        self.mnist = torchvision.datasets.MNIST(root=root,
                                                train=True,
                                                download=True,
                                                transform=transform)
        assert inlier in self.mnist.targets, "Error: inlier indice should be a valid target value"
        self.inlier = inlier
        self.indices = torch.where(self.mnist.targets == self.inlier )[0]

    def __getitem__(self, index):
        mnist_index = self.indices[index]
        return self.mnist[mnist_index][0], self.mnist[mnist_index][1]
    def __len__(self):
        return len(self.indices)

class OutlierDataset(Dataset):
    def __init__(self,root, inlier, transform):
        self.mnist = torchvision.datasets.MNIST(root=root,
                                                train=False,
                                                download=True,
                                                transform=transform)
        assert inlier in self.mnist.targets, "Error: inlier indice should be a valid target value"
        self.inlier = inlier
        self.indices = torch.where(self.mnist.targets != self.inlier)[0]
        portion = int(.20 * len(self.indices))
        self.shuffle = torch.randperm(len(self.indices))[:portion]
        self.indices = self.indices[self.shuffle]
    def __getitem__(self, index):
        mnist_index = self.indices[index]
        return self.mnist[mnist_index][0], self.mnist[mnist_index][1]
    def __len__(self):
        return len(self.indices)


transform = transforms.ToTensor()

def get_datasets():
    inlier_dataset = InlierDataset(root=ROOT, inlier=0, transform=transform)
    train_dataset = torch.utils.data.DataLoader(inlier_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    outlier_dataset = OutlierDataset(root=ROOT, inlier=0, transform=transform)
    test_dataset = torch.utils.data.DataLoader(outlier_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    return train_dataset, test_dataset