import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import numpy as np
from typing import Tuple, Optional, Union, List, Dict
from PIL import Image

# Default data directory
ROOT = os.environ.get("DATA_DIR", "./data")

class InlierDataset(Dataset):
    """
    Dataset of inlier examples for anomaly detection.
    
    This dataset contains only examples from the specified inlier class.
    """
    def __init__(self, 
                 root: str, 
                 inlier: int = 0, 
                 transform=None, 
                 dataset_type: str = "mnist",
                 train: bool = True,
                 image_size: int = 28):
        """
        Initialize the inlier dataset.
        
        Args:
            root: Root directory for data
            inlier: Index of inlier class (e.g., 0-9 for MNIST)
            transform: Transformations to apply to images
            dataset_type: Type of dataset (mnist, fashion_mnist, cifar10)
            train: Whether to use training or test set
            image_size: Size of images after resizing
        """
        self.root = root
        self.inlier = inlier
        self.train = train
        self.dataset_type = dataset_type
        self.image_size = image_size
        
        # Default transform if none provided
        if transform is None:
            if dataset_type in ["mnist", "fashion_mnist"]:
                self.transform = transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                ])
            else:  # CIFAR10 or others
                self.transform = transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                ])
        else:
            self.transform = transform
        
        # Load the appropriate dataset
        self.dataset = self._load_dataset()
        
        # Filter for inlier class
        self._filter_inlier()
        
    def _load_dataset(self):
        """Load the appropriate dataset based on dataset_type."""
        if self.dataset_type == "mnist":
            return torchvision.datasets.MNIST(
                root=self.root,
                train=self.train,
                download=True,
                transform=self.transform
            )
        elif self.dataset_type == "fashion_mnist":
            return torchvision.datasets.FashionMNIST(
                root=self.root,
                train=self.train,
                download=True,
                transform=self.transform
            )
        elif self.dataset_type == "cifar10":
            return torchvision.datasets.CIFAR10(
                root=self.root,
                train=self.train,
                download=True,
                transform=self.transform
            )
        else:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}")

    def _filter_inlier(self):
        """Filter the dataset to keep only the inlier class."""
        if hasattr(self.dataset, "targets"):
            if isinstance(self.dataset.targets, list):
                targets = torch.tensor(self.dataset.targets)
            else:
                targets = self.dataset.targets.clone().detach()
        elif hasattr(self.dataset, "labels"):
            if isinstance(self.dataset.labels, list):
                targets = torch.tensor(self.dataset.labels)
            else:
                targets = self.dataset.labels.clone().detach()
        else:
            raise AttributeError("Dataset doesn't have targets or labels attribute")
            
        assert self.inlier in targets, f"Inlier class {self.inlier} not found in dataset"
        self.indices = torch.where(targets == self.inlier)[0]
        print(f"Loaded {len(self.indices)} examples of inlier class {self.inlier}")

    def __getitem__(self, index):
        """Get item at specified index."""
        dataset_index = self.indices[index]
        return self.dataset[dataset_index]

    def __len__(self):
        """Get dataset size."""
        return len(self.indices)


class OutlierDataset(Dataset):
    """
    Dataset of outlier examples for anomaly detection.
    
    This dataset contains examples from all classes except the inlier class.
    """
    def __init__(self, 
                 root: str, 
                 inlier: int = 0, 
                 transform=None, 
                 dataset_type: str = "mnist",
                 train: bool = False,
                 image_size: int = 28,
                 outlier_portion: float = 0.2):
        """
        Initialize the outlier dataset.
        
        Args:
            root: Root directory for data
            inlier: Index of inlier class to exclude
            transform: Transformations to apply to images
            dataset_type: Type of dataset (mnist, fashion_mnist, cifar10)
            train: Whether to use training or test set
            image_size: Size of images after resizing
            outlier_portion: Portion of outliers to include in the dataset
        """
        self.root = root
        self.inlier = inlier
        self.train = train  # Usually False for outliers (test set)
        self.dataset_type = dataset_type
        self.image_size = image_size
        self.outlier_portion = outlier_portion
        
        # Default transform if none provided
        if transform is None:
            if dataset_type in ["mnist", "fashion_mnist"]:
                self.transform = transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                ])
            else:  # CIFAR10 or others
                self.transform = transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                ])
        else:
            self.transform = transform
        
        # Load the appropriate dataset
        self.dataset = self._load_dataset()
        
        # Filter for outlier classes
        self._filter_outliers()
        
    def _load_dataset(self):
        """Load the appropriate dataset based on dataset_type."""
        if self.dataset_type == "mnist":
            return torchvision.datasets.MNIST(
                root=self.root,
                train=self.train,
                download=True,
                transform=self.transform
            )
        elif self.dataset_type == "fashion_mnist":
            return torchvision.datasets.FashionMNIST(
                root=self.root,
                train=self.train,
                download=True,
                transform=self.transform
            )
        elif self.dataset_type == "cifar10":
            return torchvision.datasets.CIFAR10(
                root=self.root,
                train=self.train,
                download=True,
                transform=self.transform
            )
        else:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}")

    def _filter_outliers(self):
        """Filter the dataset to exclude the inlier class and sample a portion."""
        if hasattr(self.dataset, "targets"):
            if isinstance(self.dataset.targets, list):
                targets = torch.tensor(self.dataset.targets)
            else:
                targets = self.dataset.targets.clone().detach()
        elif hasattr(self.dataset, "labels"):
            if isinstance(self.dataset.labels, list):
                targets = torch.tensor(self.dataset.labels)
            else:
                targets = self.dataset.labels.clone().detach()
        else:
            raise AttributeError("Dataset doesn't have targets or labels attribute")
            
        # Find all non-inlier indices
        self.indices = torch.where(targets != self.inlier)[0]
        
        # Take only a portion for efficiency
        portion_size = int(self.outlier_portion * len(self.indices))
        self.shuffle = torch.randperm(len(self.indices))[:portion_size]
        self.indices = self.indices[self.shuffle]
        
        # Print dataset stats
        print(f"Loaded {len(self.indices)} examples of outlier classes (excluding {self.inlier})")
        
        # Count by class
        if len(self.indices) > 0:
            included_targets = targets[self.indices]
            classes, counts = torch.unique(included_targets, return_counts=True)
            class_distribution = {int(cls.item()): int(count.item()) for cls, count in zip(classes, counts)}
            print(f"Class distribution: {class_distribution}")

    def __getitem__(self, index):
        """Get item at specified index."""
        dataset_index = self.indices[index]
        return self.dataset[dataset_index]

    def __len__(self):
        """Get dataset size."""
        return len(self.indices)


def get_datasets(
    root: str = ROOT, 
    inlier: int = 0, 
    batch_size: int = 64,
    dataset_type: str = "mnist",
    image_size: int = 28,
    outlier_portion: float = 0.2,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    Get DataLoaders for inlier and outlier datasets.
    
    Args:
        root: Root directory for data
        inlier: Index of inlier class
        batch_size: Batch size for training
        dataset_type: Type of dataset (mnist, fashion_mnist, cifar10)
        image_size: Size of images after resizing
        outlier_portion: Portion of outliers to include
        num_workers: Number of worker threads for data loading
        
    Returns:
        Tuple of (train_dataloader, test_dataloader)
    """
    # Create transform appropriate for the dataset
    if dataset_type in ["mnist", "fashion_mnist"]:
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])
    else:  # CIFAR10 or others
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])
    
    # Create datasets
    inlier_dataset = InlierDataset(
        root=root, 
        inlier=inlier, 
        transform=transform,
        dataset_type=dataset_type,
        image_size=image_size
    )
    
    outlier_dataset = OutlierDataset(
        root=root, 
        inlier=inlier, 
        transform=transform,
        dataset_type=dataset_type,
        image_size=image_size,
        outlier_portion=outlier_portion
    )
    
    # Create data loaders
    train_dataloader = DataLoader(
        inlier_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True,
        num_workers=num_workers
    )
    
    test_dataloader = DataLoader(
        outlier_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True,
        num_workers=num_workers
    )
    
    return train_dataloader, test_dataloader