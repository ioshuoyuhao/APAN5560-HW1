"""
Model Module
Define neural network models: FCNN (MLP), CNN (SimpleCNN), EnhancedCNN, ResNet18, AssignmentCNN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class MLP(nn.Module):
    """
    Fully Connected Neural Network (Multi-Layer Perceptron) for CIFAR-10.
    Architecture: Flatten -> FC(3072, 200) -> ReLU -> FC(200, 150) -> ReLU -> FC(150, 10) -> Softmax
    """
    def __init__(self, num_classes=10):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 32 * 3, 200)
        self.fc2 = nn.Linear(200, 150)
        self.fc3 = nn.Linear(150, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=1)
        return x


class SimpleCNN(nn.Module):
    """
    Simple Convolutional Neural Network for CIFAR-10.
    Architecture: Conv2d -> ReLU -> MaxPool -> Conv2d -> ReLU -> MaxPool -> Flatten -> FC -> ReLU -> FC
    """
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # without padding output image size = (W-F)/S+1
        # output tensor dimensions: (?, 16, 32, 32)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # Input channels = 3, Output channels = 16
        # output tensor dimensions: (?, 16, 16, 16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Pooling layer
        # output tensor dimensions: (?, 32, 16, 16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # Output channels = 32
        # output tensor dimensions: (?, 32, 8, 8)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)  # Fully connected layer
        self.fc2 = nn.Linear(128, num_classes)  # Output layer for 10 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class EnhancedCNN(nn.Module):
    """
    Enhanced Convolutional Neural Network with BatchNorm and Dropout for CIFAR-10.
    Architecture: 4 Conv layers with BatchNorm -> FC with Dropout
    """
    def __init__(self, num_classes=10):
        super(EnhancedCNN, self).__init__()
        # Convolutional Layer 1 with BatchNorm
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)  # Batch Normalization after Conv1
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Layer 2 with BatchNorm
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)  # Batch Normalization after Conv2

        # Third convolutional layer
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Output channels = 64
        self.bn3 = nn.BatchNorm2d(64)  # Batch Normalization after Conv3

        # Fourth convolutional layer
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # Output channels = 128
        self.bn4 = nn.BatchNorm2d(128)  # Batch Normalization after Conv4

        # Fully connected layers with Dropout
        self.fc1 = nn.Linear(128 * 2 * 2, 128)
        self.dropout = nn.Dropout(0.5)  # Dropout with 50% probability
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # First convolutional layer
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # Conv -> BatchNorm -> ReLU -> Pool

        # Second convolutional layer
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # Conv -> BatchNorm -> ReLU -> Pool

        # Third convolutional layer
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # Conv -> BatchNorm -> ReLU -> Pool

        # Fourth convolutional layer
        x = self.pool(F.relu(self.bn4(self.conv4(x))))  # Conv -> BatchNorm -> ReLU -> Pool

        # Flatten the feature map
        x = x.view(-1, 128 * 2 * 2)

        # Fully connected layer 1 with Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        # Fully connected layer 2 (output)
        x = self.fc2(x)
        return x


class AssignmentCNN(nn.Module):
    """
    CNN Architecture for Assignment 2 Practice 1.
    Input: RGB image of size 64×64×3
    
    Architecture:
        - Conv2D: 16 filters, kernel 3×3, stride 1, padding 1
        - ReLU activation
        - MaxPooling2D: kernel 2×2, stride 2
        - Conv2D: 32 filters, kernel 3×3, stride 1, padding 1
        - ReLU activation
        - MaxPooling2D: kernel 2×2, stride 2
        - Flatten
        - Fully connected: 100 units
        - ReLU activation
        - Fully connected: 10 units (output)
    """
    def __init__(self, num_classes=10):
        super(AssignmentCNN, self).__init__()
        
        # Conv2D with 16 filters, kernel size 3×3, stride 1, padding 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        
        # MaxPooling2D with kernel size 2×2, stride 2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv2D with 32 filters, kernel size 3×3, stride 1, padding 1
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        
        # After two pooling layers: 64 -> 32 -> 16
        # Flattened size: 32 channels * 16 * 16 = 8192
        self.fc1 = nn.Linear(32 * 16 * 16, 100)  # Fully connected layer with 100 units
        
        self.fc2 = nn.Linear(100, num_classes)  # Fully connected layer with 10 units (output)

    def forward(self, x):
        # Conv Block 1: Conv -> ReLU -> MaxPool
        x = self.conv1(x)           # (batch, 3, 64, 64) -> (batch, 16, 64, 64)
        x = F.relu(x)               # ReLU activation
        x = self.pool(x)            # (batch, 16, 64, 64) -> (batch, 16, 32, 32)
        
        # Conv Block 2: Conv -> ReLU -> MaxPool
        x = self.conv2(x)           # (batch, 16, 32, 32) -> (batch, 32, 32, 32)
        x = F.relu(x)               # ReLU activation
        x = self.pool(x)            # (batch, 32, 32, 32) -> (batch, 32, 16, 16)
        
        # Flatten the output
        x = x.view(-1, 32 * 16 * 16)  # (batch, 32, 16, 16) -> (batch, 8192)
        
        # Fully connected layer with 100 units + ReLU
        x = self.fc1(x)             # (batch, 8192) -> (batch, 100)
        x = F.relu(x)               # ReLU activation
        
        # Fully connected layer with 10 units (output)
        x = self.fc2(x)             # (batch, 100) -> (batch, 10)
        
        return x


def get_model(model_name):
    """
    Define and return the appropriate model based on model_name.
    
    Args:
        model_name (str): One of 'FCNN', 'CNN', 'EnhancedCNN', 'ResNet18', or 'AssignmentCNN'.
    
    Returns:
        nn.Module: The requested neural network model.
    
    Raises:
        ValueError: If model_name is not recognized.
    """
    model_name = model_name.upper()
    
    if model_name == 'FCNN' or model_name == 'MLP':
        model = MLP()
    elif model_name == 'CNN' or model_name == 'SIMPLECNN':
        model = SimpleCNN()
    elif model_name == 'ENHANCEDCNN':
        model = EnhancedCNN()
    elif model_name == 'RESNET18':
        # Use pre-trained ResNet18, modify final layer for CIFAR-10 (10 classes)
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 10)
    elif model_name == 'ASSIGNMENTCNN':
        # Assignment 2 Practice 1: CNN for 64×64 input images
        model = AssignmentCNN()
    else:
        raise ValueError(f"Unknown model name: {model_name}. Choose from 'FCNN', 'CNN', 'EnhancedCNN', 'ResNet18', or 'AssignmentCNN'.")
    
    return model

