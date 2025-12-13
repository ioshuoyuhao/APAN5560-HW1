"""
CIFAR-10 Image Classifier Model
Assignment 2 Practice 2 - Model Deployment

This module provides the CNN classifier for image classification,
using the AssignmentCNN architecture trained on CIFAR-10.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io
import os
from typing import List, Tuple


# CIFAR-10 class names
CIFAR10_CLASSES = [
    "airplane",
    "automobile", 
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]


class AssignmentCNN(nn.Module):
    """
    CNN Architecture for Assignment 2 Practice 1.
    Input: RGB image of size 64×64×3
    """
    def __init__(self, num_classes=10):
        super(AssignmentCNN, self).__init__()
        
        # Conv2D with 16 filters, kernel size 3×3, stride 1, padding 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        
        # MaxPooling2D with kernel size 2×2, stride 2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv2D with 32 filters, kernel size 3×3, stride 1, padding 1
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        
        # Flattened size: 32 channels * 16 * 16 = 8192
        self.fc1 = nn.Linear(32 * 16 * 16, 100)
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x):
        # Conv Block 1
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Conv Block 2
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Flatten and FC layers
        x = x.view(-1, 32 * 16 * 16)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        return x


class ImageClassifier:
    """
    Image Classifier using the trained AssignmentCNN model.
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the classifier.
        
        Args:
            model_path: Path to the trained model weights (.pth file).
                       If None, uses untrained model (for demo purposes).
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AssignmentCNN(num_classes=10)
        
        # Load trained weights if available
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded model weights from {model_path}")
        else:
            print("Warning: No trained model weights found. Using untrained model.")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Define image preprocessing (resize to 64×64 as required by AssignmentCNN)
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])
    
    def preprocess_image(self, image_bytes: bytes) -> torch.Tensor:
        """
        Preprocess image bytes for model inference.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Preprocessed tensor ready for model input
        """
        # Open image from bytes
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary (handles PNG with alpha, grayscale, etc.)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        tensor = self.transform(image)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor.to(self.device)
    
    def classify(self, image_bytes: bytes) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        Classify an image.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Tuple of (predicted_class, confidence, top5_predictions)
        """
        with torch.no_grad():
            # Preprocess image
            tensor = self.preprocess_image(image_bytes)
            
            # Get model predictions
            outputs = self.model(tensor)
            
            # Apply softmax to get probabilities
            probabilities = F.softmax(outputs, dim=1)
            
            # Get top 5 predictions
            top5_prob, top5_idx = torch.topk(probabilities, 5)
            
            top5_predictions = [
                (CIFAR10_CLASSES[idx.item()], prob.item())
                for idx, prob in zip(top5_idx[0], top5_prob[0])
            ]
            
            # Get the top prediction
            predicted_idx = top5_idx[0][0].item()
            predicted_class = CIFAR10_CLASSES[predicted_idx]
            confidence = top5_prob[0][0].item()
            
        return predicted_class, confidence, top5_predictions
    
    def get_class_names(self) -> List[str]:
        """Return the list of CIFAR-10 class names."""
        return CIFAR10_CLASSES

