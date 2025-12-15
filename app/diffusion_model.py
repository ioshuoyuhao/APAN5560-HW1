"""
Diffusion Model API Wrapper for Image Generation.

This module provides a wrapper class for using the Diffusion model
to generate images via the FastAPI endpoints.
"""

import torch
import torch.nn as nn
import io
import base64
from PIL import Image
from torchvision.utils import make_grid
import math


class DiffusionImageGenerator:
    """
    Wrapper class for Diffusion-based image generation in FastAPI.
    
    Handles model initialization, training, and base64 image generation.
    """
    
    def __init__(self, image_size: int = 64, num_channels: int = 3, 
                 model_path: str = None, device: str = None):
        """
        Initialize the Diffusion image generator.
        
        Args:
            image_size: Size of generated images (default: 64)
            num_channels: Number of image channels (default: 3 for RGB)
            model_path: Optional path to load pre-trained weights
            device: Device to run on (auto-detected if None)
        """
        self.image_size = image_size
        self.num_channels = num_channels
        
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
            'Diffusion',
            image_size=image_size,
            num_channels=num_channels
        )
        self.model.to(self.device)
        
        self.is_trained = False
        
        # Load pre-trained weights if provided
        if model_path:
            self._load_weights(model_path)
    
    def _load_weights(self, model_path: str):
        """Load pre-trained model weights."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.network.load_state_dict(checkpoint['model_state_dict'])
                if 'ema_model_state_dict' in checkpoint:
                    self.model.ema_network.load_state_dict(checkpoint['ema_model_state_dict'])
                if 'normalizer_mean' in checkpoint:
                    self.model.normalizer_mean = checkpoint['normalizer_mean']
                if 'normalizer_std' in checkpoint:
                    self.model.normalizer_std = checkpoint['normalizer_std']
            else:
                self.model.network.load_state_dict(checkpoint)
            self.is_trained = True
            print(f"Loaded Diffusion model weights from {model_path}")
        except FileNotFoundError:
            print(f"Warning: No trained Diffusion model weights found at {model_path}. Using untrained model.")
        except Exception as e:
            print(f"Warning: Could not load Diffusion model weights: {e}. Using untrained model.")
    
    def train(self, train_loader, val_loader=None, epochs: int = 50, lr: float = 1e-3):
        """
        Train the Diffusion model.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation
            epochs: Number of training epochs
            lr: Learning rate
        """
        from helper_lib.trainer import train_diffusion
        
        optimizer = torch.optim.AdamW(self.model.network.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.L1Loss()
        
        # Calculate normalization statistics from training data
        self._set_normalizer(train_loader)
        
        train_diffusion(
            self.model, train_loader, criterion, optimizer,
            device=str(self.device), epochs=epochs, val_loader=val_loader
        )
        
        self.is_trained = True
    
    def _set_normalizer(self, train_loader):
        """Calculate and set normalization statistics from training data."""
        mean = torch.zeros(self.num_channels)
        std = torch.zeros(self.num_channels)
        total_samples = 0
        
        for imgs, _ in train_loader:
            batch_size = imgs.size(0)
            imgs_flat = imgs.view(batch_size, self.num_channels, -1)
            batch_mean = imgs_flat.mean(dim=(0, 2))
            batch_std = imgs_flat.std(dim=(0, 2))
            mean += batch_mean * batch_size
            std += batch_std * batch_size
            total_samples += batch_size
        
        mean /= total_samples
        std /= total_samples
        
        mean = mean.reshape(1, self.num_channels, 1, 1).to(self.device)
        std = std.reshape(1, self.num_channels, 1, 1).to(self.device)
        self.model.set_normalizer(mean, std)
    
    def generate(self, num_samples: int = 1, diffusion_steps: int = 100):
        """
        Generate images using the Diffusion model.
        
        Args:
            num_samples: Number of images to generate
            diffusion_steps: Number of diffusion steps
            
        Returns:
            torch.Tensor: Generated images in [0, 1] range
        """
        self.model.eval()
        with torch.no_grad():
            samples = self.model.generate(
                num_images=num_samples,
                diffusion_steps=diffusion_steps,
                image_size=self.image_size
            )
        return samples
    
    def generate_base64(self, num_samples: int = 1, diffusion_steps: int = 100) -> list:
        """
        Generate images and return as base64-encoded PNG strings.
        
        Args:
            num_samples: Number of images to generate
            diffusion_steps: Number of diffusion steps
            
        Returns:
            List of base64-encoded PNG strings
        """
        samples = self.generate(num_samples, diffusion_steps)
        
        images_base64 = []
        for i in range(num_samples):
            img_tensor = samples[i].cpu()
            
            # Convert to PIL Image
            if self.num_channels == 1:
                img_array = (img_tensor.squeeze().numpy() * 255).astype('uint8')
                img = Image.fromarray(img_array, mode='L')
            else:
                img_array = (img_tensor.permute(1, 2, 0).numpy() * 255).astype('uint8')
                img = Image.fromarray(img_array, mode='RGB')
            
            # Convert to base64
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            images_base64.append(img_base64)
        
        return images_base64
    
    def generate_grid(self, num_samples: int = 16, nrow: int = 4, 
                      diffusion_steps: int = 100) -> str:
        """
        Generate a grid of images and return as base64-encoded PNG.
        
        Args:
            num_samples: Number of images to generate
            nrow: Number of images per row in the grid
            diffusion_steps: Number of diffusion steps
            
        Returns:
            Base64-encoded PNG string of the grid
        """
        samples = self.generate(num_samples, diffusion_steps)
        
        # Create grid
        grid = make_grid(samples, nrow=nrow, normalize=False, padding=2)
        
        # Convert to PIL Image
        if self.num_channels == 1:
            grid_array = (grid.squeeze().cpu().numpy() * 255).astype('uint8')
            img = Image.fromarray(grid_array, mode='L')
        else:
            grid_array = (grid.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
            img = Image.fromarray(grid_array, mode='RGB')
        
        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def get_info(self) -> dict:
        """Return model information."""
        return {
            "model": "Diffusion Model",
            "image_size": self.image_size,
            "num_channels": self.num_channels,
            "is_trained": self.is_trained,
            "device": str(self.device),
            "architecture": {
                "backbone": "UNet",
                "embedding": "Sinusoidal (32 dim)",
                "schedule": "Offset Cosine"
            }
        }

