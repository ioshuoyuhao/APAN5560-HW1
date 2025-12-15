"""
MNIST GAN Model for Image Generation
Assignment 3 Practice 2 - Model Deployment

This module provides the MNIST GAN generator for handwritten digit generation,
using the MNISTGenerator architecture trained on MNIST dataset.
"""

import torch
import torch.nn as nn
from torchvision.utils import make_grid
import io
import base64
import os
from typing import List
from PIL import Image
import numpy as np


class MNISTGenerator(nn.Module):
    """
    Generator for MNIST GAN (Assignment 3).
    Generates 28x28 grayscale images from random noise vectors.
    
    Architecture:
        - Input: Noise vector of shape (BATCH_SIZE, 100)
        - Fully connected layer to 7×7×128, then reshape
        - ConvTranspose2D: 128 → 64, kernel 4, stride 2, padding 1 → 14×14
          - BatchNorm2D + ReLU
        - ConvTranspose2D: 64 → 1, kernel 4, stride 2, padding 1 → 28×28
          - Tanh activation
    """
    def __init__(self, z_dim=100):
        super(MNISTGenerator, self).__init__()
        self.z_dim = z_dim
        
        # Fully connected layer: z_dim -> 7*7*128
        self.fc = nn.Linear(z_dim, 7 * 7 * 128)
        
        # ConvTranspose2D: 128 -> 64, output size 14x14
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU(True)
        
        # ConvTranspose2D: 64 -> 1, output size 28x28
        self.deconv2 = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Flatten input if needed and pass through FC layer
        x = x.view(x.size(0), -1)  # (batch, z_dim)
        x = self.fc(x)              # (batch, 7*7*128)
        
        # Reshape to (batch, 128, 7, 7)
        x = x.view(x.size(0), 128, 7, 7)
        
        # First deconv block: 7x7 -> 14x14
        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        
        # Second deconv block: 14x14 -> 28x28
        x = self.deconv2(x)
        x = self.tanh(x)
        
        return x


class MNISTGANGenerator:
    """
    MNIST GAN Generator for API deployment.
    Generates handwritten digit images from random noise.
    """
    
    def __init__(self, model_path: str = None, z_dim: int = 100):
        """
        Initialize the generator.
        
        Args:
            model_path: Path to the trained generator weights (.pth file).
                       If None, uses untrained model (for demo purposes).
            z_dim: Dimension of latent space (noise vector). Default is 100.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.z_dim = z_dim
        self.generator = MNISTGenerator(z_dim=z_dim)
        
        # Load trained weights if available
        if model_path and os.path.exists(model_path):
            self.generator.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded generator weights from {model_path}")
        else:
            print("Warning: No trained generator weights found. Using untrained model.")
        
        self.generator.to(self.device)
        self.generator.eval()
    
    def generate(self, num_samples: int = 1) -> torch.Tensor:
        """
        Generate MNIST digit images.
        
        Args:
            num_samples: Number of images to generate.
            
        Returns:
            Tensor of shape (num_samples, 1, 28, 28) with values in [-1, 1]
        """
        with torch.no_grad():
            # Sample random noise from standard normal distribution
            noise = torch.randn(num_samples, self.z_dim).to(self.device)
            
            # Generate images
            fake_images = self.generator(noise)
            
        return fake_images
    
    def generate_base64(self, num_samples: int = 1) -> List[str]:
        """
        Generate MNIST digit images and return as base64-encoded strings.
        
        Args:
            num_samples: Number of images to generate.
            
        Returns:
            List of base64-encoded PNG images.
        """
        # Generate images
        fake_images = self.generate(num_samples)
        
        # Convert from [-1, 1] to [0, 255]
        fake_images = (fake_images + 1) / 2  # [-1, 1] -> [0, 1]
        fake_images = (fake_images * 255).clamp(0, 255).byte()
        
        base64_images = []
        for i in range(num_samples):
            # Get single image and convert to numpy
            img_tensor = fake_images[i].cpu()
            img_array = img_tensor.squeeze().numpy()
            
            # Create PIL Image
            img = Image.fromarray(img_array, mode='L')
            
            # Save to bytes buffer
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            buffer.seek(0)
            
            # Encode to base64
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            base64_images.append(img_base64)
        
        return base64_images
    
    def generate_grid(self, num_samples: int = 16, nrow: int = 4) -> str:
        """
        Generate a grid of MNIST digit images and return as base64-encoded string.
        
        Args:
            num_samples: Number of images to generate.
            nrow: Number of images per row in the grid.
            
        Returns:
            Base64-encoded PNG image of the grid.
        """
        # Generate images
        fake_images = self.generate(num_samples)
        
        # Convert from [-1, 1] to [0, 1] for make_grid
        fake_images = (fake_images + 1) / 2
        
        # Create grid - make_grid outputs (3, H, W) even for grayscale
        grid = make_grid(fake_images, nrow=nrow, normalize=False, padding=2)
        
        # Convert to numpy array: (C, H, W) -> (H, W, C)
        # make_grid always outputs 3 channels, so we permute and take first channel for grayscale
        grid_array = grid.cpu().numpy()
        
        # grid_array shape is (3, H, W) - take first channel for grayscale
        grid_array = grid_array[0]  # Now shape is (H, W)
        
        # Scale to [0, 255]
        grid_array = (grid_array * 255).astype(np.uint8)
        
        # Create PIL Image (grayscale)
        img = Image.fromarray(grid_array, mode='L')
        
        # Save to bytes buffer
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        
        # Encode to base64
        grid_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return grid_base64
    
    def get_latent_dim(self) -> int:
        """Return the latent space dimension."""
        return self.z_dim

