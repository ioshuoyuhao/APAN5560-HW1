"""
Energy-Based Model (EBM) API Wrapper for Image Generation.

This module provides a wrapper class for using the Energy-Based Model
to generate images via the FastAPI endpoints.
"""

import torch
import torch.nn as nn
import io
import base64
from PIL import Image
from torchvision.utils import make_grid
import math


class EBMImageGenerator:
    """
    Wrapper class for EBM-based image generation in FastAPI.
    
    Handles model initialization, training, and base64 image generation.
    Uses Langevin dynamics for sampling low-energy states.
    Supports both grayscale (MNIST) and RGB (CIFAR-10) images.
    """
    
    def __init__(self, image_size: int = 32, num_channels: int = 1,
                 model_path: str = None, device: str = None, 
                 steps: int = 60, step_size: float = 10.0, noise: float = 0.005):
        """
        Initialize the EBM image generator.
        
        Args:
            image_size: Size of generated images (default: 32)
            num_channels: Number of image channels (1 for grayscale/MNIST, 3 for RGB/CIFAR-10)
            model_path: Optional path to load pre-trained weights
            device: Device to run on (auto-detected if None)
            steps: Number of Langevin steps for training
            step_size: Langevin step size
            noise: Noise level for Langevin sampling
        """
        self.image_size = image_size
        self.num_channels = num_channels
        self.steps = steps
        self.step_size = step_size
        self.noise = noise
        
        # Auto-detect device
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device('mps')
            elif torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        # Import from helper_lib
        from helper_lib.model import get_model
        
        # Initialize model
        self.model = get_model(
            'EBM',
            device=str(self.device),
            image_size=image_size,
            num_channels=num_channels,
            steps=steps,
            step_size=step_size,
            noise=noise
        )
        self.model.to(self.device)
        self.model.device = self.device
        
        self.is_trained = False
        
        # Load pre-trained weights if provided
        if model_path:
            self._load_weights(model_path)
    
    def _load_weights(self, model_path: str):
        """Load pre-trained model weights."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.model.load_state_dict(checkpoint['model_state_dict'])
                # Restore hyperparameters if available
                if 'steps' in checkpoint:
                    self.model.steps = checkpoint['steps']
                if 'step_size' in checkpoint:
                    self.model.step_size = checkpoint['step_size']
                if 'noise' in checkpoint:
                    self.model.noise = checkpoint['noise']
            else:
                self.model.model.load_state_dict(checkpoint)
            self.is_trained = True
            print(f"Loaded EBM model weights from {model_path}")
        except FileNotFoundError:
            print(f"Warning: No trained EBM model weights found at {model_path}. Using untrained model.")
        except Exception as e:
            print(f"Warning: Could not load EBM model weights: {e}. Using untrained model.")
    
    def train(self, train_loader, val_loader=None, epochs: int = 10, lr: float = 0.0001):
        """
        Train the EBM model.
        
        Args:
            train_loader: DataLoader for training data (should be MNIST-style 32x32)
            val_loader: Optional DataLoader for validation
            epochs: Number of training epochs
            lr: Learning rate
        """
        from helper_lib.trainer import train_ebm
        
        optimizer = torch.optim.Adam(
            self.model.model.parameters(), 
            lr=lr, 
            betas=(0.0, 0.999)
        )
        
        train_ebm(
            self.model, train_loader, optimizer,
            device=str(self.device), epochs=epochs, val_loader=val_loader
        )
        
        self.is_trained = True
    
    def generate(self, num_samples: int = 1, steps: int = 256, 
                 step_size: float = 10.0, noise_std: float = 0.01):
        """
        Generate images using Langevin dynamics.
        
        Args:
            num_samples: Number of images to generate
            steps: Number of Langevin sampling steps (more = better quality)
            step_size: Langevin step size
            noise_std: Noise standard deviation for sampling
            
        Returns:
            torch.Tensor: Generated images in [0, 1] range
        """
        self.model.model.eval()
        samples = self.model.generate(
            num_samples=num_samples,
            steps=steps,
            step_size=step_size,
            noise_std=noise_std
        )
        return samples
    
    def generate_base64(self, num_samples: int = 1, steps: int = 256,
                        step_size: float = 10.0, noise_std: float = 0.01) -> list:
        """
        Generate images and return as base64-encoded PNG strings.
        
        Args:
            num_samples: Number of images to generate
            steps: Number of Langevin sampling steps
            step_size: Langevin step size
            noise_std: Noise standard deviation
            
        Returns:
            List of base64-encoded PNG strings
        """
        samples = self.generate(num_samples, steps, step_size, noise_std)
        
        images_base64 = []
        for i in range(num_samples):
            img_tensor = samples[i].cpu()
            
            if self.num_channels == 1:
                # Grayscale images (MNIST-style)
                img_array = (img_tensor.squeeze().numpy() * 255).astype('uint8')
                img = Image.fromarray(img_array, mode='L')
            else:
                # RGB images (CIFAR-10 style)
                # img_tensor shape: (C, H, W) -> (H, W, C)
                img_array = (img_tensor.permute(1, 2, 0).numpy() * 255).astype('uint8')
                img = Image.fromarray(img_array, mode='RGB')
            
            # Convert to base64
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            images_base64.append(img_base64)
        
        return images_base64
    
    def generate_grid(self, num_samples: int = 16, nrow: int = 4,
                      steps: int = 256, step_size: float = 10.0, 
                      noise_std: float = 0.01) -> str:
        """
        Generate a grid of images and return as base64-encoded PNG.
        
        Args:
            num_samples: Number of images to generate
            nrow: Number of images per row in the grid
            steps: Number of Langevin sampling steps
            step_size: Langevin step size
            noise_std: Noise standard deviation
            
        Returns:
            Base64-encoded PNG string of the grid
        """
        samples = self.generate(num_samples, steps, step_size, noise_std)
        
        # Create grid
        grid = make_grid(samples, nrow=nrow, normalize=False, padding=2)
        
        if self.num_channels == 1:
            # Grayscale grid
            grid_array = (grid.squeeze().cpu().numpy() * 255).astype('uint8')
            img = Image.fromarray(grid_array, mode='L')
        else:
            # RGB grid - shape: (C, H, W) -> (H, W, C)
            grid_array = (grid.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
            img = Image.fromarray(grid_array, mode='RGB')
        
        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def get_info(self) -> dict:
        """Return model information."""
        color_mode = "RGB (CIFAR-10 style)" if self.num_channels == 3 else "Grayscale (MNIST style)"
        return {
            "model": "Energy-Based Model (EBM)",
            "image_size": self.image_size,
            "num_channels": self.num_channels,
            "color_mode": color_mode,
            "is_trained": self.is_trained,
            "device": str(self.device),
            "sampling": {
                "method": "Langevin Dynamics",
                "default_steps": 256,
                "step_size": self.step_size,
                "noise": self.noise
            },
            "architecture": {
                "type": "CNN Energy Function",
                "activation": "Swish",
                "layers": [
                    f"Conv2d: {self.num_channels} → 16, k=5, s=2, p=2",
                    "Conv2d: 16 → 32, k=3, s=2, p=1",
                    "Conv2d: 32 → 64, k=3, s=2, p=1",
                    "Conv2d: 64 → 64, k=3, s=2, p=1",
                    "Flatten + Linear: 256 → 64 → 1"
                ],
                "output": "Scalar energy value"
            },
            "training": {
                "method": "Contrastive Divergence",
                "loss": "real_energy - fake_energy + regularization"
            }
        }

