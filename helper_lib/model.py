"""
Model Module
Define neural network models: FCNN (MLP), CNN (SimpleCNN), EnhancedCNN, ResNet18, AssignmentCNN, VAE, GAN, MNISTGAN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


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


def get_model(model_name, **kwargs):
    """
    Define and return the appropriate model based on model_name.
    
    Args:
        model_name (str): One of 'FCNN', 'CNN', 'EnhancedCNN', 'ResNet18', 'AssignmentCNN', 
                         'VAE', 'GAN', or 'MNISTGAN'.
        **kwargs: Additional arguments for specific models:
            - latent_dim (int): For VAE, dimension of latent space. Default is 2.
            - z_dim (int): For GAN/MNISTGAN, dimension of latent space. Default is 100.
    
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
    else:
        raise ValueError(f"Unknown model name: {model_name}. Choose from 'FCNN', 'CNN', 'EnhancedCNN', 'ResNet18', 'AssignmentCNN', 'VAE', 'GAN', or 'MNISTGAN'.")
    
    return model

