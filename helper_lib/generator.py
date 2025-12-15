"""
Generator Module
Generate samples from trained VAE and GAN models.
"""

import torch
import matplotlib.pyplot as plt
import math
from torchvision.utils import make_grid


def generate_samples(model, device, num_samples=10, latent_dim=2):
    """
    Generate samples from a trained VAE model by sampling from the latent space.
    
    Args:
        model (nn.Module): Trained VAE model with a decoder attribute.
        device (str): Device to run inference on ('cpu', 'cuda', or 'mps').
        num_samples (int): Number of samples to generate. Default is 10.
        latent_dim (int): Dimension of latent space. Default is 2.
    
    Returns:
        numpy.ndarray: Generated samples as numpy array.
    """
    model.eval()
    
    with torch.no_grad():
        # Sample from standard normal distribution in latent space
        z = torch.randn(num_samples, latent_dim).to(device)
        # Decode samples to image space
        samples = model.decoder(z)
        samples = samples.cpu().numpy()
    
    # Calculate grid dimensions
    ncols = min(6, num_samples)
    nrows = math.ceil(num_samples / ncols)
    
    # Plot samples on a grid
    fig, axes = plt.subplots(nrows, ncols, figsize=(2 * ncols, 2 * nrows))
    
    # Handle case when only one row
    if nrows == 1:
        axes = [axes] if ncols == 1 else axes
    
    for i, ax in enumerate(axes.flat if hasattr(axes, 'flat') else [axes]):
        if i < num_samples:
            ax.imshow(samples[i].squeeze(), cmap='gray')
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return samples


def generate_gan_samples(model, device, num_samples=64):
    """
    Generate samples from a trained GAN model by sampling from the latent space.
    
    Args:
        model (nn.Module): Trained GAN model with a generator attribute and z_dim.
        device (str): Device to run inference on ('cpu', 'cuda', or 'mps').
        num_samples (int): Number of samples to generate. Default is 64.
    
    Returns:
        torch.Tensor: Generated samples as tensor.
    """
    model.eval()
    z_dim = model.z_dim
    
    with torch.no_grad():
        # Sample from standard normal distribution in latent space
        noise = torch.randn(num_samples, z_dim, 1, 1).to(device)
        # Generate images using the generator
        fake = model.generator(noise).detach().cpu()
    
    # Create a grid of generated images
    grid = make_grid(fake, normalize=True, nrow=int(math.sqrt(num_samples)))
    
    # Plot the grid
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.permute(1, 2, 0))
    plt.title("Generated Samples")
    plt.axis("off")
    plt.show()
    
    return fake


def generate_mnist_gan_samples(model, device, num_samples=64, show_plot=True):
    """
    Generate samples from a trained MNIST GAN model by sampling from the latent space.
    
    Args:
        model (nn.Module): Trained MNISTGAN model with a generator attribute and z_dim.
        device (str): Device to run inference on ('cpu', 'cuda', or 'mps').
        num_samples (int): Number of samples to generate. Default is 64.
        show_plot (bool): Whether to display the generated samples. Default is True.
    
    Returns:
        torch.Tensor: Generated samples as tensor of shape (num_samples, 1, 28, 28).
    """
    model.eval()
    z_dim = model.z_dim
    
    with torch.no_grad():
        # Sample from standard normal distribution in latent space
        # MNIST generator expects flat noise vector (batch, z_dim)
        noise = torch.randn(num_samples, z_dim).to(device)
        # Generate images using the generator
        fake = model.generator(noise).detach().cpu()
    
    if show_plot:
        # Create a grid of generated images
        # Rescale from [-1, 1] to [0, 1] for visualization
        fake_rescaled = (fake + 1) / 2
        grid = make_grid(fake_rescaled, normalize=False, nrow=int(math.sqrt(num_samples)))
        
        # Plot the grid
        plt.figure(figsize=(10, 10))
        plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap='gray')
        plt.title("Generated MNIST Samples")
        plt.axis("off")
        plt.show()
    
    return fake

