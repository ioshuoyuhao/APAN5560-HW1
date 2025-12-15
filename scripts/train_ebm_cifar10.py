"""
Train Energy-Based Model (EBM) on CIFAR-10 Dataset

This script trains an Energy-Based Model on the CIFAR-10 dataset
for image generation using Langevin dynamics sampling.

Usage:
    python scripts/train_ebm_cifar10.py

The trained model will be saved to models/ebm_cifar10_best.pth
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.optim as optim

from helper_lib.data_loader import get_data_loader
from helper_lib.model import get_model
from helper_lib.trainer import train_ebm
from helper_lib.generator import generate_ebm_samples


def main():
    # ============== Configuration ==============
    DATA_DIR = "./data"
    MODEL_DIR = "./models"
    CHECKPOINT_DIR = "./models/ebm_cifar10_checkpoints"
    
    # Training hyperparameters
    BATCH_SIZE = 64
    EPOCHS = 30  # EBM training can be slow due to Langevin sampling
    LEARNING_RATE = 1e-4
    IMAGE_SIZE = 32  # CIFAR-10 native size
    NUM_CHANNELS = 3  # RGB for CIFAR-10
    
    # EBM-specific hyperparameters
    LANGEVIN_STEPS = 60
    STEP_SIZE = 10.0
    NOISE = 0.005
    ALPHA = 0.1  # Regularization coefficient
    
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
    
    # Load CIFAR-10 with normalization to [-1, 1]
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
    print(f"Number of channels: {NUM_CHANNELS} (RGB)")
    
    # ============== Model Setup ==============
    print("\n" + "="*60)
    print("Initializing Energy-Based Model...")
    print("="*60)
    
    model = get_model(
        'EBM',
        device=str(device),
        image_size=IMAGE_SIZE,
        num_channels=NUM_CHANNELS,
        alpha=ALPHA,
        steps=LANGEVIN_STEPS,
        step_size=STEP_SIZE,
        noise=NOISE
    )
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.model.parameters())
    trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"\nEBM Hyperparameters:")
    print(f"  Langevin steps: {LANGEVIN_STEPS}")
    print(f"  Step size: {STEP_SIZE}")
    print(f"  Noise: {NOISE}")
    print(f"  Alpha (regularization): {ALPHA}")
    
    # ============== Optimizer ==============
    # EBM uses Adam with specific beta values for stability
    optimizer = optim.Adam(
        model.model.parameters(), 
        lr=LEARNING_RATE, 
        betas=(0.0, 0.999)
    )
    
    print(f"\nOptimizer: Adam with lr={LEARNING_RATE}, betas=(0.0, 0.999)")
    print(f"Loss: Contrastive Divergence")
    
    # ============== Training ==============
    print("\n" + "="*60)
    print("Starting Training...")
    print("="*60)
    print("Note: EBM training involves Langevin dynamics sampling at each step,")
    print("      which can be slow. Please be patient.")
    
    trained_model = train_ebm(
        model=model,
        data_loader=train_loader,
        optimizer=optimizer,
        device=str(device),
        epochs=EPOCHS,
        val_loader=val_loader,
        checkpoint_dir=CHECKPOINT_DIR
    )
    
    # ============== Save Final Model ==============
    final_model_path = os.path.join(MODEL_DIR, "ebm_cifar10_best.pth")
    torch.save({
        'model_state_dict': trained_model.model.state_dict(),
        'image_size': IMAGE_SIZE,
        'num_channels': NUM_CHANNELS,
        'steps': LANGEVIN_STEPS,
        'step_size': STEP_SIZE,
        'noise': NOISE,
        'alpha': ALPHA,
    }, final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")
    
    # ============== Generate Samples ==============
    print("\n" + "="*60)
    print("Generating Sample Images...")
    print("="*60)
    
    samples = generate_ebm_samples(
        model=trained_model,
        device=str(device),
        num_samples=16,
        steps=256,  # More steps for better quality
        step_size=STEP_SIZE,
        noise_std=0.01,
        show_plot=True
    )
    
    print("\nTraining complete!")
    print(f"Model saved to: {final_model_path}")
    print("You can now use the /ebm/cifar10/* API endpoints.")


if __name__ == "__main__":
    main()

