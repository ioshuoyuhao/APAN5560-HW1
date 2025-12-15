"""
Train MNIST GAN
Assignment 3 - Generative Adversarial Networks

This script trains the MNISTGAN model on MNIST dataset
to generate handwritten digits using the helper library.

Architecture (as specified in Assignment 3):
    Generator:
        - Input: Noise vector of shape (BATCH_SIZE, 100)
        - Fully connected layer to 7×7×128, then reshape
        - ConvTranspose2D: 128 → 64, kernel 4, stride 2, padding 1 → 14×14
          - BatchNorm2D + ReLU
        - ConvTranspose2D: 64 → 1, kernel 4, stride 2, padding 1 → 28×28
          - Tanh activation

    Discriminator:
        - Input: Image of shape (1, 28, 28)
        - Conv2D: 1 → 64, kernel 4, stride 2, padding 1 → 14×14
          - LeakyReLU(0.2)
        - Conv2D: 64 → 128, kernel 4, stride 2, padding 1 → 7×7
          - BatchNorm2D + LeakyReLU(0.2)
        - Flatten and Linear layer to get single output (real/fake probability)
"""
import sys
import os

# Add project root to path so we can import helper_lib from the parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from helper_lib.data_loader import get_mnist_data_loader
from helper_lib.trainer import train_mnist_gan
from helper_lib.model import get_model
from helper_lib.utils import save_model
from helper_lib.generator import generate_mnist_gan_samples


def main():
    # ============================================
    # Configuration
    # ============================================
    DATA_DIR = './data'
    BATCH_SIZE = 128
    Z_DIM = 100  # Latent space dimension
    LEARNING_RATE = 0.0002
    BETA1 = 0.5  # Beta1 for Adam optimizer
    EPOCHS = 20  # Number of training epochs
    
    # Device selection
    if torch.backends.mps.is_available():
        DEVICE = 'mps'  # Apple Silicon GPU
    elif torch.cuda.is_available():
        DEVICE = 'cuda'  # NVIDIA GPU
    else:
        DEVICE = 'cpu'
    
    print("=" * 60)
    print("Assignment 3: MNIST GAN Training")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    
    # ============================================
    # Step 1: Load MNIST Data
    # ============================================
    print("\n[Step 1] Loading MNIST dataset...")
    
    train_loader = get_mnist_data_loader(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        train=True,
        normalize_to_minus_one=True  # Normalize to [-1, 1] for Tanh output
    )
    
    print(f"  Training samples: {len(train_loader.dataset)}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Number of batches: {len(train_loader)}")
    
    # ============================================
    # Step 2: Create MNIST GAN Model
    # ============================================
    print("\n[Step 2] Creating MNISTGAN model...")
    
    model = get_model("MNISTGAN", z_dim=Z_DIM)
    
    # Print model architecture
    print("\nGenerator Architecture:")
    print(model.generator)
    print("\nDiscriminator Architecture:")
    print(model.discriminator)
    
    # Count parameters
    gen_params = sum(p.numel() for p in model.generator.parameters())
    disc_params = sum(p.numel() for p in model.discriminator.parameters())
    print(f"\n  Generator parameters: {gen_params:,}")
    print(f"  Discriminator parameters: {disc_params:,}")
    print(f"  Total parameters: {gen_params + disc_params:,}")
    
    # ============================================
    # Step 3: Train the GAN
    # ============================================
    print("\n[Step 3] Training MNIST GAN...")
    print("-" * 60)
    print(f"  Epochs: {EPOCHS}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Beta1: {BETA1}")
    print("-" * 60)
    
    trained_model = train_mnist_gan(
        model=model,
        data_loader=train_loader,
        device=DEVICE,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        beta1=BETA1
    )
    
    # ============================================
    # Step 4: Generate Sample Images
    # ============================================
    print("\n[Step 4] Generating sample images...")
    
    # Generate and display samples
    samples = generate_mnist_gan_samples(
        model=trained_model,
        device=DEVICE,
        num_samples=64,
        show_plot=True
    )
    print(f"  Generated {samples.shape[0]} samples of shape {samples.shape[1:]}")
    
    # ============================================
    # Step 5: Save the Model
    # ============================================
    print("\n[Step 5] Saving trained model...")
    
    # Create models directory if it doesn't exist
    os.makedirs('./models', exist_ok=True)
    
    # Save the full model
    model_path = './models/mnist_gan.pth'
    save_model(trained_model, model_path)
    
    # Also save just the generator (for deployment)
    generator_path = './models/mnist_gan_generator.pth'
    torch.save(trained_model.generator.state_dict(), generator_path)
    print(f"  Generator saved to: {generator_path}")
    
    # ============================================
    # Summary
    # ============================================
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"  Model saved to: {model_path}")
    print(f"  Generator saved to: {generator_path}")
    print(f"  Total epochs trained: {EPOCHS}")
    print("=" * 60)
    print("\nYou can now use the trained model in the FastAPI application!")


if __name__ == "__main__":
    main()

