"""
Model Module
Define neural network models: FCNN (MLP), CNN (SimpleCNN), EnhancedCNN, ResNet18, AssignmentCNN, VAE, GAN, MNISTGAN, Diffusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
import math
import copy


class MLP(nn.Module):
    """
    Fully Connected Neural Network (Multi-Layer Perceptron) for CIFAR-10.
    Architecture: Flatten -> FC(3072, 200) -> ReLU -> FC(200, 150) -> ReLU -> FC(150, 10) -> Softmax
    """
    def __init__(self, num_classes=10):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 32 * 3, 200)
        self.fc2 = nn.Linear(200, 150)
        self.fc3 = nn.Linear(150, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=1)
        return x


class SimpleCNN(nn.Module):
    """
    Simple Convolutional Neural Network for CIFAR-10.
    Architecture: Conv2d -> ReLU -> MaxPool -> Conv2d -> ReLU -> MaxPool -> Flatten -> FC -> ReLU -> FC
    """
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # without padding output image size = (W-F)/S+1
        # output tensor dimensions: (?, 16, 32, 32)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # Input channels = 3, Output channels = 16
        # output tensor dimensions: (?, 16, 16, 16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Pooling layer
        # output tensor dimensions: (?, 32, 16, 16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # Output channels = 32
        # output tensor dimensions: (?, 32, 8, 8)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)  # Fully connected layer
        self.fc2 = nn.Linear(128, num_classes)  # Output layer for 10 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class EnhancedCNN(nn.Module):
    """
    Enhanced Convolutional Neural Network with BatchNorm and Dropout for CIFAR-10.
    Architecture: 4 Conv layers with BatchNorm -> FC with Dropout
    """
    def __init__(self, num_classes=10):
        super(EnhancedCNN, self).__init__()
        # Convolutional Layer 1 with BatchNorm
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)  # Batch Normalization after Conv1
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Layer 2 with BatchNorm
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)  # Batch Normalization after Conv2

        # Third convolutional layer
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Output channels = 64
        self.bn3 = nn.BatchNorm2d(64)  # Batch Normalization after Conv3

        # Fourth convolutional layer
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # Output channels = 128
        self.bn4 = nn.BatchNorm2d(128)  # Batch Normalization after Conv4

        # Fully connected layers with Dropout
        self.fc1 = nn.Linear(128 * 2 * 2, 128)
        self.dropout = nn.Dropout(0.5)  # Dropout with 50% probability
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # First convolutional layer
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # Conv -> BatchNorm -> ReLU -> Pool

        # Second convolutional layer
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # Conv -> BatchNorm -> ReLU -> Pool

        # Third convolutional layer
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # Conv -> BatchNorm -> ReLU -> Pool

        # Fourth convolutional layer
        x = self.pool(F.relu(self.bn4(self.conv4(x))))  # Conv -> BatchNorm -> ReLU -> Pool

        # Flatten the feature map
        x = x.view(-1, 128 * 2 * 2)

        # Fully connected layer 1 with Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        # Fully connected layer 2 (output)
        x = self.fc2(x)
        return x


class AssignmentCNN(nn.Module):
    """
    CNN Architecture for Assignment 2 Practice 1.
    Input: RGB image of size 64×64×3
    
    Architecture:
        - Conv2D: 16 filters, kernel 3×3, stride 1, padding 1
        - ReLU activation
        - MaxPooling2D: kernel 2×2, stride 2
        - Conv2D: 32 filters, kernel 3×3, stride 1, padding 1
        - ReLU activation
        - MaxPooling2D: kernel 2×2, stride 2
        - Flatten
        - Fully connected: 100 units
        - ReLU activation
        - Fully connected: 10 units (output)
    """
    def __init__(self, num_classes=10):
        super(AssignmentCNN, self).__init__()
        
        # Conv2D with 16 filters, kernel size 3×3, stride 1, padding 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        
        # MaxPooling2D with kernel size 2×2, stride 2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv2D with 32 filters, kernel size 3×3, stride 1, padding 1
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        
        # After two pooling layers: 64 -> 32 -> 16
        # Flattened size: 32 channels * 16 * 16 = 8192
        self.fc1 = nn.Linear(32 * 16 * 16, 100)  # Fully connected layer with 100 units
        
        self.fc2 = nn.Linear(100, num_classes)  # Fully connected layer with 10 units (output)

    def forward(self, x):
        # Conv Block 1: Conv -> ReLU -> MaxPool
        x = self.conv1(x)           # (batch, 3, 64, 64) -> (batch, 16, 64, 64)
        x = F.relu(x)               # ReLU activation
        x = self.pool(x)            # (batch, 16, 64, 64) -> (batch, 16, 32, 32)
        
        # Conv Block 2: Conv -> ReLU -> MaxPool
        x = self.conv2(x)           # (batch, 16, 32, 32) -> (batch, 32, 32, 32)
        x = F.relu(x)               # ReLU activation
        x = self.pool(x)            # (batch, 32, 32, 32) -> (batch, 32, 16, 16)
        
        # Flatten the output
        x = x.view(-1, 32 * 16 * 16)  # (batch, 32, 16, 16) -> (batch, 8192)
        
        # Fully connected layer with 100 units + ReLU
        x = self.fc1(x)             # (batch, 8192) -> (batch, 100)
        x = F.relu(x)               # ReLU activation
        
        # Fully connected layer with 10 units (output)
        x = self.fc2(x)             # (batch, 100) -> (batch, 10)
        
        return x


# ============================================================================
# Variational Autoencoder (VAE) Components
# ============================================================================

class VariationalEncoder(nn.Module):
    """
    Variational Encoder for VAE.
    Maps input image to mean and log variance of a latent distribution.
    Architecture: 3 Conv layers -> Flatten -> FC to mu and logvar
    Input: 32x32 grayscale image (1 channel)
    """
    def __init__(self, latent_dim=2):
        super(VariationalEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 4, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class VAEDecoder(nn.Module):
    """
    Decoder for VAE.
    Maps latent vector back to image space.
    Architecture: FC -> Reshape -> 3 ConvTranspose layers
    Output: 32x32 grayscale image (1 channel)
    """
    def __init__(self, latent_dim=2):
        super(VAEDecoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 128 * 4 * 4)
        self.convtrans1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.convtrans2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.convtrans3 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 128, 4, 4)
        x = F.relu(self.convtrans1(x))
        x = F.relu(self.convtrans2(x))
        x = torch.sigmoid(self.convtrans3(x))
        return x


class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) for image generation.
    Combines VariationalEncoder and VAEDecoder with reparameterization trick.
    
    Args:
        latent_dim (int): Dimension of latent space. Default is 2.
    """
    def __init__(self, latent_dim=2):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = VariationalEncoder(latent_dim)
        self.decoder = VAEDecoder(latent_dim)

    def reparameterize(self, mu, logvar):
        """Reparameterization trick to sample from N(mu, var) from N(0,1)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decoder(z)
        return reconstruction, mu, logvar


# ============================================================================
# Generative Adversarial Network (GAN) Components
# ============================================================================

class Critic(nn.Module):
    """
    Critic (Discriminator) for WGAN.
    Evaluates whether an image is real or generated.
    Architecture: 5 Conv layers with BatchNorm and LeakyReLU
    Input: 64x64 RGB image (3 channels)
    Output: Single value (critic score)
    """
    def __init__(self):
        super(Critic, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(128)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(256)
        self.act3 = nn.LeakyReLU(0.2, inplace=True)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchnorm4 = nn.BatchNorm2d(512)
        self.act4 = nn.LeakyReLU(0.2, inplace=True)

        self.conv5 = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.act3(x)

        x = self.conv4(x)
        x = self.batchnorm4(x)
        x = self.act4(x)

        x = self.conv5(x)
        x = self.flatten(x)

        return x


class Generator(nn.Module):
    """
    Generator for WGAN.
    Generates images from random noise vectors in latent space.
    Architecture: 5 ConvTranspose layers with BatchNorm and ReLU
    Input: Random noise vector of size z_dim
    Output: 64x64 RGB image (3 channels)
    
    Args:
        z_dim (int): Dimension of latent space. Default is 100.
    """
    def __init__(self, z_dim=100):
        super(Generator, self).__init__()
        self.z_dim = z_dim

        self.deconv1 = nn.ConvTranspose2d(z_dim, 512, kernel_size=4, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(512, momentum=0.9)
        self.act1 = nn.ReLU(True)

        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256, momentum=0.9)
        self.act2 = nn.ReLU(True)

        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128, momentum=0.9)
        self.act3 = nn.ReLU(True)

        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(64, momentum=0.9)
        self.act4 = nn.ReLU(True)

        self.deconv5 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Reshape input to (batch, z_dim, 1, 1)
        x = x.view(x.size(0), self.z_dim, 1, 1)

        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.deconv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        x = self.deconv3(x)
        x = self.bn3(x)
        x = self.act3(x)

        x = self.deconv4(x)
        x = self.bn4(x)
        x = self.act4(x)

        x = self.deconv5(x)
        x = self.tanh(x)

        return x


class GAN(nn.Module):
    """
    Generative Adversarial Network (GAN) wrapper.
    Combines Generator and Critic (Discriminator) for image generation.
    
    Args:
        z_dim (int): Dimension of latent space. Default is 100.
    """
    def __init__(self, z_dim=100):
        super(GAN, self).__init__()
        self.z_dim = z_dim
        self.generator = Generator(z_dim)
        self.critic = Critic()

    def forward(self, z):
        """Generate images from latent vectors."""
        return self.generator(z)


# ============================================================================
# MNIST GAN Components (Assignment 3)
# ============================================================================

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
    
    Args:
        z_dim (int): Dimension of latent space. Default is 100.
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


class MNISTDiscriminator(nn.Module):
    """
    Discriminator for MNIST GAN (Assignment 3).
    Evaluates whether a 28x28 grayscale image is real or generated.
    
    Architecture:
        - Input: Image of shape (1, 28, 28)
        - Conv2D: 1 → 64, kernel 4, stride 2, padding 1 → 14×14
          - LeakyReLU(0.2)
        - Conv2D: 64 → 128, kernel 4, stride 2, padding 1 → 7×7
          - BatchNorm2D + LeakyReLU(0.2)
        - Flatten and Linear layer to get single output (real/fake probability)
    """
    def __init__(self):
        super(MNISTDiscriminator, self).__init__()
        
        # Conv2D: 1 -> 64, output size 14x14
        self.conv1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        
        # Conv2D: 64 -> 128, output size 7x7
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)
        
        # Flatten and Linear: 7*7*128 -> 1
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(7 * 7 * 128, 1)

    def forward(self, x):
        # First conv block: 28x28 -> 14x14
        x = self.conv1(x)
        x = self.act1(x)
        
        # Second conv block: 14x14 -> 7x7
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        
        # Flatten and output
        x = self.flatten(x)  # (batch, 7*7*128)
        x = self.fc(x)       # (batch, 1)
        
        return x


class MNISTGAN(nn.Module):
    """
    MNIST GAN wrapper (Assignment 3).
    Combines MNISTGenerator and MNISTDiscriminator for MNIST digit generation.
    
    Args:
        z_dim (int): Dimension of latent space. Default is 100.
    """
    def __init__(self, z_dim=100):
        super(MNISTGAN, self).__init__()
        self.z_dim = z_dim
        self.generator = MNISTGenerator(z_dim)
        self.discriminator = MNISTDiscriminator()

    def forward(self, z):
        """Generate images from latent vectors."""
        return self.generator(z)


# ============================================================================
# Diffusion Model Components (Module 8)
# ============================================================================

def linear_diffusion_schedule(diffusion_times, min_rate=1e-4, max_rate=0.02):
    """
    Linear diffusion schedule.
    
    Args:
        diffusion_times: Tensor of shape (T,) with values in [0, 1)
        min_rate: Minimum beta value
        max_rate: Maximum beta value
    
    Returns:
        noise_rates: Tensor of shape (T,)
        signal_rates: Tensor of shape (T,)
    """
    diffusion_times = diffusion_times.to(dtype=torch.float32)
    betas = min_rate + diffusion_times * (max_rate - min_rate)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    signal_rates = torch.sqrt(alpha_bars)
    noise_rates = torch.sqrt(1.0 - alpha_bars)
    return noise_rates, signal_rates


def cosine_diffusion_schedule(diffusion_times):
    """
    Cosine diffusion schedule.
    Starts clear (slow noise growth), then rapidly blurs near the end.
    
    Args:
        diffusion_times: Tensor with values in [0, 1]
    
    Returns:
        noise_rates, signal_rates
    """
    signal_rates = torch.cos(diffusion_times * math.pi / 2)
    noise_rates = torch.sin(diffusion_times * math.pi / 2)
    return noise_rates, signal_rates


def offset_cosine_diffusion_schedule(diffusion_times, min_signal_rate=0.02, max_signal_rate=0.95):
    """
    Offset cosine diffusion schedule. Avoids extreme values.
    
    Args:
        diffusion_times: Tensor with values in [0, 1]
        min_signal_rate: Minimum signal rate
        max_signal_rate: Maximum signal rate
    
    Returns:
        noise_rates, signal_rates
    """
    original_shape = diffusion_times.shape
    diffusion_times_flat = diffusion_times.flatten()

    start_angle = torch.acos(torch.tensor(max_signal_rate, dtype=torch.float32, device=diffusion_times.device))
    end_angle = torch.acos(torch.tensor(min_signal_rate, dtype=torch.float32, device=diffusion_times.device))

    diffusion_angles = start_angle + diffusion_times_flat * (end_angle - start_angle)

    signal_rates = torch.cos(diffusion_angles).reshape(original_shape)
    noise_rates = torch.sin(diffusion_angles).reshape(original_shape)

    return noise_rates, signal_rates


class SinusoidalEmbedding(nn.Module):
    """
    Sinusoidal embedding for encoding noise variance in diffusion models.
    """
    def __init__(self, num_frequencies=16):
        super().__init__()
        self.num_frequencies = num_frequencies
        frequencies = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), num_frequencies))
        self.register_buffer("angular_speeds", 2.0 * math.pi * frequencies.view(1, 1, 1, -1))

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, 1, 1, 1)
        Returns:
            Tensor of shape (B, 1, 1, 2 * num_frequencies)
        """
        x = x.expand(-1, 1, 1, self.num_frequencies)
        sin_part = torch.sin(self.angular_speeds * x)
        cos_part = torch.cos(self.angular_speeds * x)
        return torch.cat([sin_part, cos_part], dim=-1)


class DiffusionResidualBlock(nn.Module):
    """Residual block for UNet in diffusion models."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.needs_projection = in_channels != out_channels
        if self.needs_projection:
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.proj = nn.Identity()

        self.norm = nn.BatchNorm2d(in_channels, affine=False)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, x):
        residual = self.proj(x)
        x = self.swish(self.conv1(x))
        x = self.conv2(x)
        return x + residual


class DiffusionDownBlock(nn.Module):
    """Down-sampling block for UNet in diffusion models."""
    def __init__(self, width, block_depth, in_channels):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(block_depth):
            self.blocks.append(DiffusionResidualBlock(in_channels, width))
            in_channels = width
        self.pool = nn.AvgPool2d(kernel_size=2)

    def forward(self, x, skips):
        for block in self.blocks:
            x = block(x)
            skips.append(x)
        x = self.pool(x)
        return x


class DiffusionUpBlock(nn.Module):
    """Up-sampling block for UNet in diffusion models."""
    def __init__(self, width, block_depth, in_channels):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(block_depth):
            self.blocks.append(DiffusionResidualBlock(in_channels + width, width))
            in_channels = width

    def forward(self, x, skips):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        for block in self.blocks:
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            x = block(x)
        return x


class UNet(nn.Module):
    """
    UNet architecture for diffusion models.
    
    Args:
        image_size: Size of input images (assumes square)
        num_channels: Number of image channels (1 for grayscale, 3 for RGB)
        embedding_dim: Dimension of noise embedding
    """
    def __init__(self, image_size, num_channels, embedding_dim=32):
        super().__init__()
        self.initial = nn.Conv2d(num_channels, 32, kernel_size=1)
        self.num_channels = num_channels
        self.image_size = image_size
        self.embedding_dim = embedding_dim
        self.embedding = SinusoidalEmbedding(num_frequencies=16)
        self.embedding_proj = nn.Conv2d(embedding_dim, 32, kernel_size=1)

        self.down1 = DiffusionDownBlock(32, in_channels=64, block_depth=2)
        self.down2 = DiffusionDownBlock(64, in_channels=32, block_depth=2)
        self.down3 = DiffusionDownBlock(96, in_channels=64, block_depth=2)

        self.mid1 = DiffusionResidualBlock(in_channels=96, out_channels=128)
        self.mid2 = DiffusionResidualBlock(in_channels=128, out_channels=128)

        self.up1 = DiffusionUpBlock(96, in_channels=128, block_depth=2)
        self.up2 = DiffusionUpBlock(64, block_depth=2, in_channels=96)
        self.up3 = DiffusionUpBlock(32, block_depth=2, in_channels=64)

        self.final = nn.Conv2d(32, num_channels, kernel_size=1)
        nn.init.zeros_(self.final.weight)

    def forward(self, noisy_images, noise_variances):
        skips = []
        x = self.initial(noisy_images)
        noise_emb = self.embedding(noise_variances)
        noise_emb = F.interpolate(noise_emb.permute(0, 3, 1, 2), size=(self.image_size, self.image_size), mode='nearest')
        x = torch.cat([x, noise_emb], dim=1)

        x = self.down1(x, skips)
        x = self.down2(x, skips)
        x = self.down3(x, skips)

        x = self.mid1(x)
        x = self.mid2(x)

        x = self.up1(x, skips)
        x = self.up2(x, skips)
        x = self.up3(x, skips)

        return self.final(x)


class DiffusionModel(nn.Module):
    """
    Diffusion Model wrapper with training and generation capabilities.
    
    Args:
        model: UNet backbone network
        schedule_fn: Diffusion schedule function (e.g., offset_cosine_diffusion_schedule)
    """
    def __init__(self, model, schedule_fn=None):
        super().__init__()
        self.network = model
        # EMA network for more stable sampling
        self.ema_network = copy.deepcopy(model)
        self.ema_network.eval()
        self.ema_decay = 0.8
        self.schedule_fn = schedule_fn if schedule_fn else offset_cosine_diffusion_schedule
        self.normalizer_mean = 0.0
        self.normalizer_std = 1.0
        self.image_size = model.image_size
        self.num_channels = model.num_channels

    def to(self, device):
        super().to(device)
        self.ema_network.to(device)
        return self

    def set_normalizer(self, mean, std):
        """Set normalization parameters for denormalization during generation."""
        self.normalizer_mean = mean
        self.normalizer_std = std

    def denormalize(self, x):
        """Denormalize images back to [0, 1] range."""
        return torch.clamp(x * self.normalizer_std + self.normalizer_mean, 0.0, 1.0)

    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        """Denoise images using the network."""
        if training:
            network = self.network
            network.train()
        else:
            network = self.ema_network
            network.eval()

        pred_noises = network(noisy_images, noise_rates ** 2)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates
        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, diffusion_steps):
        """Generate images by reversing the diffusion process."""
        step_size = 1.0 / diffusion_steps
        current_images = initial_noise

        for step in range(diffusion_steps):
            t = torch.ones((initial_noise.shape[0], 1, 1, 1), device=initial_noise.device) * (1 - step * step_size)
            noise_rates, signal_rates = self.schedule_fn(t)
            pred_noises, pred_images = self.denoise(current_images, noise_rates, signal_rates, training=False)

            next_diffusion_times = t - step_size
            next_noise_rates, next_signal_rates = self.schedule_fn(next_diffusion_times)
            current_images = next_signal_rates * pred_images + next_noise_rates * pred_noises

        return pred_images

    def generate(self, num_images, diffusion_steps, image_size=None, initial_noise=None):
        """Generate new images from random noise."""
        if image_size is None:
            image_size = self.image_size
        if initial_noise is None:
            initial_noise = torch.randn(
                (num_images, self.num_channels, image_size, image_size),
                device=next(self.parameters()).device
            )
        with torch.no_grad():
            return self.denormalize(self.reverse_diffusion(initial_noise, diffusion_steps))

    def train_step(self, images, optimizer, loss_fn):
        """Single training step."""
        images = (images - self.normalizer_mean) / self.normalizer_std
        noises = torch.randn_like(images)

        diffusion_times = torch.rand((images.size(0), 1, 1, 1), device=images.device)
        noise_rates, signal_rates = self.schedule_fn(diffusion_times)
        noisy_images = signal_rates * images + noise_rates * noises

        pred_noises, _ = self.denoise(noisy_images, noise_rates, signal_rates, training=True)
        loss = loss_fn(pred_noises, noises)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update EMA network
        with torch.no_grad():
            for ema_param, param in zip(self.ema_network.parameters(), self.network.parameters()):
                ema_param.copy_(self.ema_decay * ema_param + (1. - self.ema_decay) * param)

        return loss.item()

    def test_step(self, images, loss_fn):
        """Single validation step."""
        images = (images - self.normalizer_mean) / self.normalizer_std
        noises = torch.randn_like(images)

        diffusion_times = torch.rand((images.size(0), 1, 1, 1), device=images.device)
        noise_rates, signal_rates = self.schedule_fn(diffusion_times)
        noisy_images = signal_rates * images + noise_rates * noises

        with torch.no_grad():
            pred_noises, _ = self.denoise(noisy_images, noise_rates, signal_rates, training=False)
            loss = loss_fn(pred_noises, noises)

        return loss.item()


# ============================================================================
# Energy-Based Model (EBM) Components (For Assignment 4)
# ============================================================================

def swish(x):
    """Swish activation function - smooth alternative to ReLU."""
    return x * torch.sigmoid(x)


class EnergyModel(nn.Module):
    """
    Energy-Based Model neural network.
    Maps input images to a scalar energy value.
    Uses swish activation for smooth gradients during Langevin sampling.
    
    Args:
        num_channels: Number of input channels (1 for grayscale, 3 for RGB)
        image_size: Size of input images (default: 32)
    
    Output: Scalar energy value
    """
    def __init__(self, num_channels=1, image_size=32):
        super(EnergyModel, self).__init__()
        self.num_channels = num_channels
        self.image_size = image_size
        
        self.conv1 = nn.Conv2d(num_channels, 16, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        
        self.flatten = nn.Flatten()
        # Calculate flattened size based on image_size (after 4 stride-2 convolutions: size/16)
        final_size = image_size // 16
        self.fc1 = nn.Linear(64 * final_size * final_size, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = swish(self.conv1(x))
        x = swish(self.conv2(x))
        x = swish(self.conv3(x))
        x = swish(self.conv4(x))
        x = self.flatten(x)
        x = swish(self.fc1(x))
        return self.fc2(x)


class EBMBuffer:
    """
    Sample buffer for efficient EBM training.
    Stores generated samples to avoid starting from scratch each time.
    """
    def __init__(self, model, device, buffer_size=128, image_size=32, num_channels=1):
        self.model = model
        self.device = device
        self.image_size = image_size
        self.num_channels = num_channels
        # Start with random images in the buffer
        self.examples = [
            torch.rand((1, num_channels, image_size, image_size), device=device) * 2 - 1 
            for _ in range(buffer_size)
        ]
    
    def sample_new_examples(self, batch_size, generate_fn, steps, step_size, noise_std):
        """Sample new examples using Langevin dynamics."""
        import random
        import numpy as np
        
        n_new = np.random.binomial(batch_size, 0.05)  # ~5% new random images
        
        # Generate new random images
        new_rand_imgs = torch.rand(
            (n_new, self.num_channels, self.image_size, self.image_size), device=self.device
        ) * 2 - 1
        
        # Sample old images from buffer
        old_imgs = torch.cat(random.choices(self.examples, k=batch_size - n_new), dim=0)
        
        inp_imgs = torch.cat([new_rand_imgs, old_imgs], dim=0)
        
        # Run Langevin dynamics
        new_imgs = generate_fn(self.model, inp_imgs, steps, step_size, noise_std)
        
        # Update buffer
        self.examples = list(torch.split(new_imgs, 1, dim=0)) + self.examples
        self.examples = self.examples[:8192]  # Cap buffer size
        
        return new_imgs


class EBM(nn.Module):
    """
    Energy-Based Model wrapper with training capabilities.
    Uses Contrastive Divergence loss for training.
    
    Args:
        model: EnergyModel neural network
        alpha: Regularization coefficient
        steps: Number of Langevin sampling steps
        step_size: Langevin step size
        noise: Noise level for sampling
        device: Device to run on
        image_size: Size of images
        num_channels: Number of image channels (1 for grayscale, 3 for RGB)
    """
    def __init__(self, model, alpha=0.1, steps=60, step_size=10, noise=0.005, 
                 device='cpu', image_size=32, num_channels=1):
        super().__init__()
        self.device = device
        self.model = model
        self.alpha = alpha
        self.steps = steps
        self.step_size = step_size
        self.noise = noise
        self.image_size = image_size
        self.num_channels = num_channels
        self.buffer = None  # Will be initialized when training starts
        
    def _init_buffer(self):
        """Initialize the sample buffer."""
        if self.buffer is None:
            self.buffer = EBMBuffer(
                self.model, self.device, buffer_size=128, 
                image_size=self.image_size, num_channels=self.num_channels
            )
    
    def _generate_samples_langevin(self, nn_energy_model, inp_imgs, steps, step_size, noise_std):
        """Generate samples using Langevin dynamics."""
        nn_energy_model.eval()
        
        for _ in range(steps):
            with torch.no_grad():
                noise = torch.randn_like(inp_imgs) * noise_std
                inp_imgs = (inp_imgs + noise).clamp(-1.0, 1.0)
            
            inp_imgs.requires_grad_(True)
            
            # Compute energy and gradients
            energy = nn_energy_model(inp_imgs)
            grads, = torch.autograd.grad(
                energy, inp_imgs, grad_outputs=torch.ones_like(energy)
            )
            
            # Apply gradient descent with clipping
            with torch.no_grad():
                grads = grads.clamp(-0.03, 0.03)
                inp_imgs = (inp_imgs - step_size * grads).clamp(-1.0, 1.0)
        
        return inp_imgs.detach()
    
    def generate(self, num_samples, steps=None, step_size=None, noise_std=None):
        """Generate new images from random noise."""
        if steps is None:
            steps = self.steps * 4  # More steps for generation
        if step_size is None:
            step_size = self.step_size
        if noise_std is None:
            noise_std = self.noise
        
        self.model.eval()
        x = torch.rand(
            (num_samples, self.num_channels, self.image_size, self.image_size), 
            device=self.device
        ) * 2 - 1
        
        # Note: Do NOT use torch.no_grad() here - Langevin dynamics requires gradients
        samples = self._generate_samples_langevin(
            self.model, x, steps, step_size, noise_std
        )
        
        # Scale from [-1, 1] to [0, 1]
        return torch.clamp((samples + 1) / 2, 0, 1)
    
    def train_step(self, real_imgs, optimizer):
        """Single training step using Contrastive Divergence."""
        self._init_buffer()
        self.model.train()
        
        # Add noise to real images
        real_imgs = real_imgs + torch.randn_like(real_imgs) * self.noise
        real_imgs = torch.clamp(real_imgs, -1.0, 1.0)
        
        # Sample fake images from buffer using Langevin dynamics
        fake_imgs = self.buffer.sample_new_examples(
            batch_size=real_imgs.size(0),
            generate_fn=self._generate_samples_langevin,
            steps=self.steps,
            step_size=self.step_size,
            noise_std=self.noise
        )
        
        # Combine and compute energy
        inp_imgs = torch.cat([real_imgs, fake_imgs], dim=0)
        inp_imgs = inp_imgs.clone().detach().to(self.device).requires_grad_(False)
        
        out_scores = self.model(inp_imgs)
        real_out, fake_out = torch.split(
            out_scores, [real_imgs.size(0), fake_imgs.size(0)], dim=0
        )
        
        # Contrastive Divergence loss: minimize energy on real, maximize on fake
        cdiv_loss = real_out.mean() - fake_out.mean()
        reg_loss = self.alpha * (real_out.pow(2).mean() + fake_out.pow(2).mean())
        loss = cdiv_loss + reg_loss
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
        optimizer.step()
        
        return {
            'loss': loss.item(),
            'cdiv_loss': cdiv_loss.item(),
            'reg_loss': reg_loss.item(),
            'real_energy': real_out.mean().item(),
            'fake_energy': fake_out.mean().item()
        }
    
    def test_step(self, real_imgs):
        """Validation step."""
        self.model.eval()
        batch_size = real_imgs.shape[0]
        fake_imgs = torch.rand(
            (batch_size, self.num_channels, self.image_size, self.image_size), 
            device=self.device
        ) * 2 - 1
        
        inp_imgs = torch.cat([real_imgs, fake_imgs], dim=0)
        
        with torch.no_grad():
            out_scores = self.model(inp_imgs)
            real_out, fake_out = torch.split(out_scores, batch_size, dim=0)
            cdiv = real_out.mean() - fake_out.mean()
        
        return {
            'cdiv_loss': cdiv.item(),
            'real_energy': real_out.mean().item(),
            'fake_energy': fake_out.mean().item()
        }


def get_model(model_name, **kwargs):
    """
    Define and return the appropriate model based on model_name.
    
    Args:
        model_name (str): One of 'FCNN', 'CNN', 'EnhancedCNN', 'ResNet18', 'AssignmentCNN', 
                         'VAE', 'GAN', 'MNISTGAN', 'Diffusion', or 'EBM'.
        **kwargs: Additional arguments for specific models:
            - latent_dim (int): For VAE, dimension of latent space. Default is 2.
            - z_dim (int): For GAN/MNISTGAN, dimension of latent space. Default is 100.
            - image_size (int): For Diffusion/EBM, size of input images. Default is 64/32.
            - num_channels (int): For Diffusion/EBM, number of image channels. Default is 3/1.
            - embedding_dim (int): For Diffusion, noise embedding dimension. Default is 32.
            - schedule_fn: For Diffusion, diffusion schedule function. Default is offset_cosine.
            - alpha (float): For EBM, regularization coefficient. Default is 0.1.
            - steps (int): For EBM, Langevin sampling steps. Default is 60.
            - step_size (float): For EBM, Langevin step size. Default is 10.
            - noise (float): For EBM, noise level. Default is 0.005.
            - device (str): For EBM, device to run on. Default is 'cpu'.
    
    Returns:
        nn.Module: The requested neural network model.
    
    Raises:
        ValueError: If model_name is not recognized.
    """
    model_name = model_name.upper()
    
    if model_name == 'FCNN' or model_name == 'MLP':
        model = MLP()
    elif model_name == 'CNN' or model_name == 'SIMPLECNN':
        model = SimpleCNN()
    elif model_name == 'ENHANCEDCNN':
        model = EnhancedCNN()
    elif model_name == 'RESNET18':
        # Use pre-trained ResNet18, modify final layer for CIFAR-10 (10 classes)
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 10)
    elif model_name == 'ASSIGNMENTCNN':
        # Assignment 2 Practice 1: CNN for 64×64 input images
        model = AssignmentCNN()
    elif model_name == 'VAE':
        # Variational Autoencoder for image generation
        latent_dim = kwargs.get('latent_dim', 2)
        model = VAE(latent_dim=latent_dim)
    elif model_name == 'GAN':
        # Generative Adversarial Network for CelebA image generation
        z_dim = kwargs.get('z_dim', 100)
        model = GAN(z_dim=z_dim)
    elif model_name == 'MNISTGAN':
        # MNIST GAN for handwritten digit generation (Assignment 3)
        z_dim = kwargs.get('z_dim', 100)
        model = MNISTGAN(z_dim=z_dim)
    elif model_name == 'DIFFUSION':
        # Diffusion Model for image generation (Module 8)
        image_size = kwargs.get('image_size', 64)
        num_channels = kwargs.get('num_channels', 3)
        embedding_dim = kwargs.get('embedding_dim', 32)
        schedule_fn = kwargs.get('schedule_fn', offset_cosine_diffusion_schedule)
        unet = UNet(image_size, num_channels, embedding_dim)
        model = DiffusionModel(unet, schedule_fn)
    elif model_name == 'EBM' or model_name == 'ENERGYMODEL':
        # Energy-Based Model for image generation (Module 8)
        device = kwargs.get('device', 'cpu')
        alpha = kwargs.get('alpha', 0.1)
        steps = kwargs.get('steps', 60)
        step_size = kwargs.get('step_size', 10)
        noise = kwargs.get('noise', 0.005)
        image_size = kwargs.get('image_size', 32)
        num_channels = kwargs.get('num_channels', 1)  # 1 for grayscale, 3 for RGB
        energy_model = EnergyModel(num_channels=num_channels, image_size=image_size)
        model = EBM(energy_model, alpha=alpha, steps=steps, step_size=step_size, 
                    noise=noise, device=device, image_size=image_size, 
                    num_channels=num_channels)
    else:
        raise ValueError(f"Unknown model name: {model_name}. Choose from 'FCNN', 'CNN', 'EnhancedCNN', 'ResNet18', 'AssignmentCNN', 'VAE', 'GAN', 'MNISTGAN', 'Diffusion', or 'EBM'.")
    
    return model

