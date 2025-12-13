"""
Utils Module
Utility functions for neural network projects.
"""

import torch
import os


def save_model(model, path):
    """
    Save the model's state dictionary to a file.
    
    Args:
        model (nn.Module): The neural network model to save.
        path (str): File path to save the model.
    
    Returns:
        None
    """
    # Create directory if it doesn't exist
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    # Save model state dictionary
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(model, path, device='cpu'):
    """
    Load model weights from a file.
    
    Args:
        model (nn.Module): The neural network model (architecture must match saved weights).
        path (str): File path to load the model from.
        device (str): Device to load the model onto. Default is 'cpu'.
    
    Returns:
        nn.Module: The model with loaded weights.
    """
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    print(f"Model loaded from {path}")
    return model

