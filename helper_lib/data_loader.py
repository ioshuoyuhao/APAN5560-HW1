"""
Data Loader Module
Encapsulates data loading logic for CIFAR-10 dataset.
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_data_loader(data_dir, batch_size=32, train=True, image_size=None):
    """
    Create and return a data loader for CIFAR-10 dataset.
    
    Args:
        data_dir (str): Directory to store/load the dataset.
        batch_size (int): Number of samples per batch. Default is 32.
        train (bool): If True, load training set; otherwise load test set. Default is True.
        image_size (int or tuple, optional): Resize images to this size. 
            If None, uses original CIFAR-10 size (32×32).
            For AssignmentCNN, use image_size=64.
    
    Returns:
        DataLoader: PyTorch DataLoader for the specified dataset.
    """
    # Build transform list
    transform_list = []
    
    # Add resize transform if image_size is specified
    if image_size is not None:
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        transform_list.append(transforms.Resize(image_size))
    
    # Always convert to tensor
    transform_list.append(transforms.ToTensor())
    
    # Compose transforms
    transform = transforms.Compose(transform_list)
    
    # Load CIFAR-10 dataset
    dataset = datasets.CIFAR10(
        root=data_dir,
        train=train,
        download=True,
        transform=transform
    )
    
    # Create and return DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train  # Shuffle only for training data
    )
    
    return loader
