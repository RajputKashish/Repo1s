"""
Data Loading and Preprocessing Module for PPCM-X

This module handles dataset loading, preprocessing, and batching for both
plaintext training and encrypted inference pipelines.

Supported Datasets:
    - MNIST (28x28 grayscale)
    - CIFAR-10 (32x32 RGB)
    - Fashion-MNIST (28x28 grayscale)
"""

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from typing import Tuple, Optional, Dict, Any
import numpy as np
import os


class HECompatibleTransform:
    """
    Transform pipeline optimized for HE-compatible inference.
    Normalizes inputs to [-1, 1] range for polynomial activation stability.
    """
    
    def __init__(self, dataset_name: str = "mnist"):
        self.dataset_name = dataset_name.lower()
        self._setup_transforms()
    
    def _setup_transforms(self):
        """Configure transforms based on dataset."""
        if self.dataset_name in ["mnist", "fashion_mnist"]:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1]
            ])
            self.input_shape = (1, 28, 28)
            self.num_classes = 10
            
        elif self.dataset_name == "cifar10":
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            self.input_shape = (3, 32, 32)
            self.num_classes = 10
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
    
    def __call__(self, x):
        return self.transform(x)


class EncryptedInferenceDataset(Dataset):
    """
    Dataset wrapper for encrypted inference with pre-flattening support.
    """
    
    def __init__(self, base_dataset: Dataset, flatten: bool = False):
        self.base_dataset = base_dataset
        self.flatten = flatten
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        x, y = self.base_dataset[idx]
        if self.flatten:
            x = x.view(-1)
        return x, y


def get_dataset(
    dataset_name: str = "mnist",
    data_dir: str = "./data",
    train: bool = True,
    download: bool = True
) -> Dataset:
    """
    Load dataset with HE-compatible preprocessing.
    
    Args:
        dataset_name: Name of dataset ('mnist', 'cifar10', 'fashion_mnist')
        data_dir: Directory to store/load data
        train: If True, load training set; else test set
        download: If True, download dataset if not present
    
    Returns:
        PyTorch Dataset object
    """
    transform = HECompatibleTransform(dataset_name)
    
    dataset_map = {
        "mnist": datasets.MNIST,
        "cifar10": datasets.CIFAR10,
        "fashion_mnist": datasets.FashionMNIST,
    }
    
    if dataset_name.lower() not in dataset_map:
        raise ValueError(f"Dataset {dataset_name} not supported. "
                        f"Choose from: {list(dataset_map.keys())}")
    
    dataset_class = dataset_map[dataset_name.lower()]
    
    return dataset_class(
        root=data_dir,
        train=train,
        download=download,
        transform=transform
    )


def get_data_loaders(
    dataset_name: str = "mnist",
    batch_size: int = 64,
    num_workers: int = 4,
    data_dir: str = "./data",
    val_split: float = 0.1,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        dataset_name: Name of dataset
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        data_dir: Directory for data storage
        val_split: Fraction of training data for validation
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load datasets
    train_dataset = get_dataset(dataset_name, data_dir, train=True)
    test_dataset = get_dataset(dataset_name, data_dir, train=False)
    
    # Split training into train/val
    num_train = len(train_dataset)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    
    split_idx = int(np.floor(val_split * num_train))
    train_indices, val_indices = indices[split_idx:], indices[:split_idx]
    
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(train_dataset, val_indices)
    
    # Create data loaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def get_sample_batch(
    dataset_name: str = "mnist",
    batch_size: int = 1,
    flatten: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get a sample batch for testing/demonstration.
    
    Args:
        dataset_name: Name of dataset
        batch_size: Number of samples
        flatten: If True, flatten input tensors
    
    Returns:
        Tuple of (inputs, labels)
    """
    dataset = get_dataset(dataset_name, train=False)
    
    if flatten:
        dataset = EncryptedInferenceDataset(dataset, flatten=True)
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    inputs, labels = next(iter(loader))
    
    return inputs, labels


def get_dataset_info(dataset_name: str) -> Dict[str, Any]:
    """
    Get metadata about a dataset.
    
    Args:
        dataset_name: Name of dataset
    
    Returns:
        Dictionary with dataset information
    """
    transform = HECompatibleTransform(dataset_name)
    
    info = {
        "name": dataset_name,
        "input_shape": transform.input_shape,
        "num_classes": transform.num_classes,
        "input_size": np.prod(transform.input_shape),
    }
    
    # Add dataset-specific info
    if dataset_name.lower() == "mnist":
        info["description"] = "Handwritten digit classification (0-9)"
        info["train_size"] = 60000
        info["test_size"] = 10000
    elif dataset_name.lower() == "cifar10":
        info["description"] = "Natural image classification (10 classes)"
        info["train_size"] = 50000
        info["test_size"] = 10000
    elif dataset_name.lower() == "fashion_mnist":
        info["description"] = "Fashion item classification (10 classes)"
        info["train_size"] = 60000
        info["test_size"] = 10000
    
    return info


if __name__ == "__main__":
    # Test data loading
    print("Testing data loader module...")
    
    for dataset_name in ["mnist", "cifar10"]:
        print(f"\n--- {dataset_name.upper()} ---")
        info = get_dataset_info(dataset_name)
        print(f"Input shape: {info['input_shape']}")
        print(f"Num classes: {info['num_classes']}")
        
        # Test sample batch
        x, y = get_sample_batch(dataset_name, batch_size=4)
        print(f"Sample batch shape: {x.shape}")
        print(f"Labels: {y.tolist()}")
        print(f"Value range: [{x.min():.3f}, {x.max():.3f}]")
    
    print("\nData loader tests passed!")
