"""
Practice 1: CNN Architecture
Assignment 2 - Convolutional Neural Networks

This script trains the AssignmentCNN model on CIFAR-10 dataset
using the helper library we built.

Architecture:
    - Input: RGB image of size 64×64×3
    - Conv2D: 16 filters, kernel 3×3, stride 1, padding 1
    - ReLU activation
    - MaxPooling2D: kernel 2×2, stride 2
    - Conv2D: 32 filters, kernel 3×3, stride 1, padding 1
    - ReLU activation
    - MaxPooling2D: kernel 2×2, stride 2
    - Flatten
    - Fully connected: 100 units + ReLU
    - Fully connected: 10 units (output)
"""
import sys
import os

# Add project root to path so we can import helper_lib from the parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Import from helper library
from helper_lib.data_loader import get_data_loader
from helper_lib.trainer import train_model
from helper_lib.evaluator import evaluate_model
from helper_lib.model import get_model
from helper_lib.utils import save_model

import torch.nn as nn
import torch.optim as optim


def main():
    # ============================================
    # Configuration
    # ============================================
    DATA_DIR = './data'
    BATCH_SIZE = 64
    IMAGE_SIZE = 64  # Resize CIFAR-10 images from 32×32 to 64×64
    LEARNING_RATE = 0.001
    EPOCHS = 10
    DEVICE = 'cpu'  # Change to 'cuda' if GPU is available
    
    print("=" * 60)
    print("Practice 1: CNN Architecture - Assignment 2")
    print("=" * 60)
    
    # ============================================
    # Step 1: Load Data
    # ============================================
    print("\n[Step 1] Loading CIFAR-10 dataset (resized to 64×64)...")
    
    train_loader = get_data_loader(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        train=True,
        image_size=IMAGE_SIZE  # Resize to 64×64
    )
    
    test_loader = get_data_loader(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        train=False,
        image_size=IMAGE_SIZE  # Resize to 64×64
    )
    
    print(f"  Training samples: {len(train_loader.dataset)}")
    print(f"  Test samples: {len(test_loader.dataset)}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Image size: {IMAGE_SIZE}×{IMAGE_SIZE}")
    
    # ============================================
    # Step 2: Create Model
    # ============================================
    print("\n[Step 2] Creating AssignmentCNN model...")
    
    model = get_model("AssignmentCNN")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # ============================================
    # Step 3: Define Loss and Optimizer
    # ============================================
    print("\n[Step 3] Setting up loss function and optimizer...")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"  Loss function: CrossEntropyLoss")
    print(f"  Optimizer: Adam (lr={LEARNING_RATE})")
    
    # ============================================
    # Step 4: Train Model
    # ============================================
    print("\n[Step 4] Training the model...")
    print("-" * 60)
    
    trained_model = train_model(
        model=model,
        data_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=DEVICE,
        epochs=EPOCHS
    )
    
    # ============================================
    # Step 5: Evaluate Model
    # ============================================
    print("\n[Step 5] Evaluating on test set...")
    print("-" * 60)
    
    avg_loss, accuracy = evaluate_model(
        model=trained_model,
        data_loader=test_loader,
        criterion=criterion,
        device=DEVICE
    )
    
    # ============================================
    # Step 6: Save Model (Optional)
    # ============================================
    print("\n[Step 6] Saving model...")
    
    save_model(trained_model, './models/assignment_cnn.pth')
    
    # ============================================
    # Summary
    # ============================================
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"  Final Test Accuracy: {accuracy:.2f}%")
    print(f"  Final Test Loss: {avg_loss:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

