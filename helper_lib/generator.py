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


def generate_diffusion_samples(model, device, num_samples=10, diffusion_steps=100, show_plot=True):
    """
    Generate samples from a trained Diffusion model by running reverse diffusion.
    
    Args:
        model (nn.Module): Trained DiffusionModel with generate method.
        device (str): Device to run inference on ('cpu', 'cuda', or 'mps').
        num_samples (int): Number of samples to generate. Default is 10.
        diffusion_steps (int): Number of diffusion steps for generation. Default is 100.
        show_plot (bool): Whether to display the generated samples. Default is True.
    
    Returns:
        torch.Tensor: Generated samples as tensor.
    """
    model.eval()
    model.to(device)
    
    # Generate num_samples points from a standard normal distribution
    # and run reverse diffusion to construct the images
    with torch.no_grad():
        samples = model.generate(
            num_images=num_samples,
            diffusion_steps=diffusion_steps
        ).cpu()
    
    if show_plot:
        # Calculate grid dimensions
        ncols = min(6, num_samples)
        nrows = math.ceil(num_samples / ncols)
        
        # Create a grid of generated images
        grid = make_grid(samples, normalize=True, nrow=ncols)
        
        # Plot the grid
        plt.figure(figsize=(2 * ncols, 2 * nrows))
        
        # Handle both grayscale and RGB images
        grid_np = grid.permute(1, 2, 0).numpy()
        if grid_np.shape[2] == 1:
            plt.imshow(grid_np.squeeze(), cmap='gray')
        else:
            plt.imshow(grid_np)
        
        plt.title(f"Generated Diffusion Samples ({diffusion_steps} steps)")
        plt.axis("off")
        plt.show()
    
    return samples


def generate_ebm_samples(model, device, num_samples=10, steps=256, step_size=10.0, 
                         noise_std=0.01, show_plot=True):
    """
    Generate samples from a trained EBM model using Langevin dynamics.
    
    Args:
        model (nn.Module): Trained EBM model with generate method.
        device (str): Device to run inference on ('cpu', 'cuda', or 'mps').
        num_samples (int): Number of samples to generate. Default is 10.
        steps (int): Number of Langevin sampling steps. Default is 256.
        step_size (float): Langevin step size. Default is 10.0.
        noise_std (float): Noise standard deviation for sampling. Default is 0.01.
        show_plot (bool): Whether to display the generated samples. Default is True.
    
    Returns:
        torch.Tensor: Generated samples as tensor (already in [0, 1] range).
    """
    model.model.eval()
    model.to(device)
    model.device = device
    
    # Generate samples using Langevin dynamics
    with torch.no_grad():
        samples = model.generate(
            num_samples=num_samples,
            steps=steps,
            step_size=step_size,
            noise_std=noise_std
        ).cpu()
    
    if show_plot:
        # Calculate grid dimensions
        ncols = min(8, num_samples)
        nrows = math.ceil(num_samples / ncols)
        
        # Create a grid of generated images
        grid = make_grid(samples, normalize=False, nrow=ncols)
        
        # Plot the grid
        plt.figure(figsize=(2 * ncols, 2 * nrows))
        
        # EBM generates grayscale images
        grid_np = grid.permute(1, 2, 0).numpy()
        if grid_np.shape[2] == 1:
            plt.imshow(grid_np.squeeze(), cmap='gray')
        else:
            plt.imshow(grid_np)
        
        plt.title(f"Generated EBM Samples ({steps} Langevin steps)")
        plt.axis("off")
        plt.show()
    
    return samples

