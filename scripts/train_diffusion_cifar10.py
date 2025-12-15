"""
Train Diffusion Model on CIFAR-10 Dataset

This script trains a Diffusion Model (UNet-based) on the CIFAR-10 dataset
for image generation. The trained model can be used via the FastAPI endpoints.

Usage:
    python scripts/train_diffusion_cifar10.py

The trained model will be saved to models/diffusion_best.pth
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim

from helper_lib.data_loader import get_data_loader
from helper_lib.model import get_model
from helper_lib.trainer import train_diffusion
from helper_lib.generator import generate_diffusion_samples


def main():
    # ============== Configuration ==============
    DATA_DIR = "./data"
    MODEL_DIR = "./models"
    CHECKPOINT_DIR = "./models/diffusion_checkpoints"
    
    # Training hyperparameters
    BATCH_SIZE = 64
    EPOCHS = 50  # Diffusion models need more epochs
    LEARNING_RATE = 1e-4
    IMAGE_SIZE = 64  # Resize CIFAR-10 to 64x64 for better quality
    NUM_CHANNELS = 3  # RGB
    EMBEDDING_DIM = 32
    
    # Create directories
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # ============== Device Setup ==============
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using CUDA GPU")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # ============== Data Loading ==============
    print("\n" + "="*60)
    print("Loading CIFAR-10 Dataset...")
    print("="*60)
    
    # Load CIFAR-10 with normalization to [-1, 1] and resize to 64x64
    train_loader = get_data_loader(
        DATA_DIR, 
        batch_size=BATCH_SIZE, 
        train=True, 
        image_size=IMAGE_SIZE,
        normalize_to_minus_one=True
    )
    
    val_loader = get_data_loader(
        DATA_DIR, 
        batch_size=BATCH_SIZE, 
        train=False, 
        image_size=IMAGE_SIZE,
        normalize_to_minus_one=True
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    
    # ============== Model Setup ==============
    print("\n" + "="*60)
    print("Initializing Diffusion Model...")
    print("="*60)
    
    model = get_model(
        'Diffusion',
        image_size=IMAGE_SIZE,
        num_channels=NUM_CHANNELS,
        embedding_dim=EMBEDDING_DIM
    )
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # ============== Optimizer & Loss ==============
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.L1Loss()  # MAE loss works well for diffusion
    
    print(f"\nOptimizer: AdamW with lr={LEARNING_RATE}")
    print(f"Loss: L1 (MAE)")
    
    # ============== Training ==============
    print("\n" + "="*60)
    print("Starting Training...")
    print("="*60)
    
    trained_model = train_diffusion(
        model=model,
        data_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=str(device),
        epochs=EPOCHS,
        val_loader=val_loader,
        checkpoint_dir=CHECKPOINT_DIR
    )
    
    # ============== Save Final Model ==============
    final_model_path = os.path.join(MODEL_DIR, "diffusion_best.pth")
    torch.save({
        'model_state_dict': trained_model.unet.state_dict(),
        'ema_state_dict': trained_model.ema_unet.state_dict(),
        'normalizer_mean': trained_model.normalizer_mean,
        'normalizer_variance': trained_model.normalizer_variance,
        'image_size': IMAGE_SIZE,
        'num_channels': NUM_CHANNELS,
        'embedding_dim': EMBEDDING_DIM,
    }, final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")
    
    # ============== Generate Samples ==============
    print("\n" + "="*60)
    print("Generating Sample Images...")
    print("="*60)
    
    samples = generate_diffusion_samples(
        model=trained_model,
        device=str(device),
        num_samples=16,
        diffusion_steps=100,
        show_plot=True
    )
    
    print("\nTraining complete!")
    print(f"Model saved to: {final_model_path}")
    print("You can now use the /diffusion/* API endpoints.")


if __name__ == "__main__":
    main()

