"""
Data Loader Module
Encapsulates data loading logic for CIFAR-10 and MNIST datasets.
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_data_loader(data_dir, batch_size=32, train=True, image_size=None, normalize_to_minus_one=False):
    """
    Create and return a data loader for CIFAR-10 dataset.
    
    Args:
        data_dir (str): Directory to store/load the dataset.
        batch_size (int): Number of samples per batch. Default is 32.
        train (bool): If True, load training set; otherwise load test set. Default is True.
        image_size (int or tuple, optional): Resize images to this size. 
            If None, uses original CIFAR-10 size (32×32).
            For AssignmentCNN, use image_size=64.
        normalize_to_minus_one (bool): If True, normalize to [-1, 1] range (for EBM/Diffusion).
                                       If False, keep in [0, 1] range. Default is False.
    
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
    
    # Normalize to [-1, 1] if requested (for EBM/Diffusion models)
    if normalize_to_minus_one:
        transform_list.append(transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))
    
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


def get_mnist_data_loader(data_dir, batch_size=64, train=True, normalize_to_minus_one=True):
    """
    Create and return a data loader for MNIST dataset.
    
    Args:
        data_dir (str): Directory to store/load the dataset.
        batch_size (int): Number of samples per batch. Default is 64.
        train (bool): If True, load training set; otherwise load test set. Default is True.
        normalize_to_minus_one (bool): If True, normalize to [-1, 1] range (for GAN with Tanh).
                                       If False, normalize to [0, 1] range. Default is True.
    
    Returns:
        DataLoader: PyTorch DataLoader for the MNIST dataset.
    """
    # Build transform list
    transform_list = [transforms.ToTensor()]
    
    # Normalize: ToTensor() scales to [0, 1], then Normalize shifts to [-1, 1]
    if normalize_to_minus_one:
        transform_list.append(transforms.Normalize([0.5], [0.5]))
    
    # Compose transforms
    transform = transforms.Compose(transform_list)
    
    # Load MNIST dataset
    dataset = datasets.MNIST(
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
