"""
Quick test script to verify EBM sample generation works.
This script loads the trained EBM model and generates sample images.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from torchvision.utils import save_image
from helper_lib import EBM, EnergyModel


def main():
    print("=" * 60)
    print("EBM Sample Generation Test")
    print("=" * 60)
    
    # Configuration
    model_path = "./models/ebm_cifar10_best.pth"
    output_path = "./models/ebm_test_samples.png"
    num_samples = 16
    num_channels = 3
    image_size = 32
    steps = 60
    
    # Create EnergyModel
    print("\n1. Creating EnergyModel...")
    energy_model = EnergyModel(num_channels=num_channels, image_size=image_size)
    
    # Create EBM wrapper
    print("2. Creating EBM wrapper...")
    model = EBM(
        model=energy_model, 
        num_channels=num_channels, 
        image_size=image_size, 
        device='cpu'
    )
    
    # Load trained weights
    print(f"3. Loading weights from {model_path}...")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model.model.load_state_dict(checkpoint['model_state_dict'])
    print("   Model loaded successfully!")
    
    # Generate samples
    print(f"\n4. Generating {num_samples} samples with {steps} Langevin steps...")
    print("   (This may take a moment...)")
    samples = model.generate(num_samples=num_samples, steps=steps)
    print(f"   Generated samples shape: {samples.shape}")
    
    # Save samples as image grid
    print(f"\n5. Saving samples to {output_path}...")
    save_image(samples, output_path, nrow=4, padding=2)
    print("   Saved!")
    
    print("\n" + "=" * 60)
    print("SUCCESS! EBM generation works correctly.")
    print(f"Check the output at: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

