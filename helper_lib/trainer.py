"""
Trainer Module
Abstract the training loop for neural network models.
"""

import torch
import torch.nn.functional as F
import torch.optim as optim


def train_model(model, data_loader, criterion, optimizer, device='cpu', epochs=10):
    """
    Run several iterations of the training loop and return the trained model.
    
    Args:
        model (nn.Module): The neural network model to train.
        data_loader (DataLoader): DataLoader for training data.
        criterion: Loss function (e.g., nn.CrossEntropyLoss()).
        optimizer: Optimizer (e.g., optim.Adam()).
        device (str): Device to train on ('cpu' or 'cuda'). Default is 'cpu'.
        epochs (int): Number of training epochs. Default is 10.
    
    Returns:
        nn.Module: The trained model.
    """
    # Move model to the specified device
    model = model.to(device)
    model.train()
    
    correct = 0
    total = 0
    
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(data_loader, 0):
            inputs, labels = data
            # Move data to device
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Compute predictions
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Compute loss
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Print statistics every 100 mini-batches
            if i % 100 == 99:
                print(f"[{epoch + 1}, {i + 1}] accuracy: {correct/total:.3f}, loss: {running_loss / 100:.3f}")
                running_loss = 0.0
    
    print("Finished Training")
    return model


def vae_loss_function(recon_x, x, mu, logvar, beta=500):
    """
    Compute the VAE loss function: beta * BCE + KL divergence.
    
    Args:
        recon_x: Reconstructed image from decoder.
        x: Original input image.
        mu: Mean from encoder.
        logvar: Log variance from encoder.
        beta (float): Weight for reconstruction loss. Default is 500.
    
    Returns:
        torch.Tensor: Total loss value.
    """
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return beta * BCE + KLD


def train_vae_model(model, data_loader, criterion, optimizer, device='cpu', epochs=10):
    """
    Run several iterations of the VAE training loop and return the trained model.
    
    Args:
        model (nn.Module): The VAE model to train.
        data_loader (DataLoader): DataLoader for training data.
        criterion: Loss function (should be vae_loss_function or similar).
        optimizer: Optimizer (e.g., optim.Adam()).
        device (str): Device to train on ('cpu', 'cuda', or 'mps'). Default is 'cpu'.
        epochs (int): Number of training epochs. Default is 10.
    
    Returns:
        nn.Module: The trained VAE model.
    """
    # Move model to the specified device
    model = model.to(device)
    model.train()
    
    for epoch in range(epochs):
        running_loss = 0.0
        batch_count = 0
        
        for i, data in enumerate(data_loader, 0):
            inputs, labels = data
            # Move data to device
            inputs = inputs.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass - VAE returns reconstruction, mu, logvar
            recon, mu, logvar = model(inputs)
            
            # Compute loss using VAE loss function
            loss = criterion(recon, inputs, mu, logvar)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            batch_count += 1
            
            # Print statistics every 100 mini-batches
            if i % 100 == 99:
                avg_loss = running_loss / batch_count
                print(f"[Epoch {epoch + 1}, Batch {i + 1}] loss: {avg_loss:.4f}")
                running_loss = 0.0
                batch_count = 0
        
        # Print epoch summary
        print(f"Epoch {epoch + 1}/{epochs} completed")
    
    print("Finished Training VAE")
    return model


def train_gan(model, data_loader, criterion, optimizer, device='cpu', epochs=10,
              n_critic=5, clip_value=0.01, lr=5e-5):
    """
    Run several iterations of the WGAN training loop and return the trained model.
    
    Args:
        model (nn.Module): The GAN model to train (with generator and critic attributes).
        data_loader (DataLoader): DataLoader for training data.
        criterion: Not used for WGAN (kept for API consistency). Pass None.
        optimizer: Not used directly (kept for API consistency). Pass None.
            Optimizers are created internally using RMSprop.
        device (str): Device to train on ('cpu', 'cuda', or 'mps'). Default is 'cpu'.
        epochs (int): Number of training epochs. Default is 10.
        n_critic (int): Number of critic updates per generator update. Default is 5.
        clip_value (float): Weight clipping value for WGAN. Default is 0.01.
        lr (float): Learning rate for optimizers. Default is 5e-5.
    
    Returns:
        nn.Module: The trained GAN model.
    """
    # Move model to the specified device
    model = model.to(device)
    gen = model.generator
    critic = model.critic
    z_dim = model.z_dim
    
    # Create optimizers for generator and critic
    opt_gen = optim.RMSprop(gen.parameters(), lr=lr)
    opt_critic = optim.RMSprop(critic.parameters(), lr=lr)
    
    gen.train()
    critic.train()
    
    for epoch in range(epochs):
        for batch_idx, (real, _) in enumerate(data_loader):
            real = real.to(device)
            batch_size = real.size(0)

            # === Train Critic === #
            for _ in range(n_critic):
                noise = torch.randn(batch_size, z_dim, 1, 1).to(device)
                fake = gen(noise).detach()
                critic_real = critic(real).mean()
                critic_fake = critic(fake).mean()
                loss_critic = -(critic_real - critic_fake)

                critic.zero_grad()
                loss_critic.backward()
                opt_critic.step()

                # Weight clipping for WGAN
                for p in critic.parameters():
                    p.data.clamp_(-clip_value, clip_value)

            # === Train Generator === #
            noise = torch.randn(batch_size, z_dim, 1, 1).to(device)
            fake = gen(noise)
            loss_gen = -critic(fake).mean()

            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            if batch_idx % 100 == 0:
                print(f"[Epoch {epoch + 1}/{epochs}] [Batch {batch_idx}/{len(data_loader)}] "
                      f"[D loss: {loss_critic.item():.4f}] [G loss: {loss_gen.item():.4f}]")
        
        # Print epoch summary
        print(f"Epoch {epoch + 1}/{epochs} completed")
    
    print("Finished Training GAN")
    return model


def train_mnist_gan(model, data_loader, device='cpu', epochs=10, lr=0.0002, beta1=0.5):
    """
    Train MNIST GAN using standard GAN training with BCE loss (Assignment 3).
    
    Args:
        model (nn.Module): The MNISTGAN model to train (with generator and discriminator attributes).
        data_loader (DataLoader): DataLoader for MNIST training data.
        device (str): Device to train on ('cpu', 'cuda', or 'mps'). Default is 'cpu'.
        epochs (int): Number of training epochs. Default is 10.
        lr (float): Learning rate for optimizers. Default is 0.0002.
        beta1 (float): Beta1 parameter for Adam optimizer. Default is 0.5.
    
    Returns:
        nn.Module: The trained MNISTGAN model.
    """
    import torch.nn as nn
    
    # Move model to the specified device
    model = model.to(device)
    generator = model.generator
    discriminator = model.discriminator
    z_dim = model.z_dim
    
    # Loss function - Binary Cross Entropy with Logits
    criterion = nn.BCEWithLogitsLoss()
    
    # Create optimizers for generator and discriminator
    opt_gen = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    opt_disc = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
    
    generator.train()
    discriminator.train()
    
    # Track losses for visualization
    g_losses = []
    d_losses = []
    
    for epoch in range(epochs):
        for batch_idx, (real_images, _) in enumerate(data_loader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            
            # Labels for real and fake images
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            
            # ========================
            # Train Discriminator
            # ========================
            opt_disc.zero_grad()
            
            # Loss on real images
            real_output = discriminator(real_images)
            d_loss_real = criterion(real_output, real_labels)
            
            # Generate fake images
            noise = torch.randn(batch_size, z_dim).to(device)
            fake_images = generator(noise).detach()
            
            # Loss on fake images
            fake_output = discriminator(fake_images)
            d_loss_fake = criterion(fake_output, fake_labels)
            
            # Total discriminator loss
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            opt_disc.step()
            
            # ========================
            # Train Generator
            # ========================
            opt_gen.zero_grad()
            
            # Generate fake images
            noise = torch.randn(batch_size, z_dim).to(device)
            fake_images = generator(noise)
            
            # Generator wants discriminator to think fake images are real
            fake_output = discriminator(fake_images)
            g_loss = criterion(fake_output, real_labels)
            
            g_loss.backward()
            opt_gen.step()
            
            # Track losses
            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())
            
            # Print statistics every 100 batches
            if batch_idx % 100 == 0:
                print(f"[Epoch {epoch + 1}/{epochs}] [Batch {batch_idx}/{len(data_loader)}] "
                      f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")
        
        # Print epoch summary
        print(f"Epoch {epoch + 1}/{epochs} completed - "
              f"Avg D loss: {sum(d_losses[-len(data_loader):]) / len(data_loader):.4f}, "
              f"Avg G loss: {sum(g_losses[-len(data_loader):]) / len(data_loader):.4f}")
    
    print("Finished Training MNIST GAN")
    return model


def train_diffusion(model, data_loader, criterion, optimizer, device='cpu', epochs=10,
                    val_loader=None, checkpoint_dir=None):
    """
    Run several iterations of the diffusion model training loop and return the trained model.
    
    Args:
        model (nn.Module): The DiffusionModel to train.
        data_loader (DataLoader): DataLoader for training data.
        criterion: Loss function (e.g., nn.L1Loss() or nn.MSELoss()).
        optimizer: Optimizer (e.g., optim.AdamW()).
        device (str): Device to train on ('cpu', 'cuda', or 'mps'). Default is 'cpu'.
        epochs (int): Number of training epochs. Default is 10.
        val_loader (DataLoader): Optional DataLoader for validation data.
        checkpoint_dir (str): Optional directory to save checkpoints.
    
    Returns:
        nn.Module: The trained DiffusionModel.
    """
    import os
    from tqdm import tqdm
    
    # Create checkpoint directory if specified
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    model.to(device)
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_losses = []
        
        # Training loop with progress bar
        loader_with_progress = tqdm(data_loader, ncols=120, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for images, _ in loader_with_progress:
            images = images.to(device)
            loss = model.train_step(images, optimizer, criterion)
            train_losses.append(loss)
            loader_with_progress.set_postfix(loss=f'{loss:.4f}')
        
        avg_train_loss = sum(train_losses) / len(train_losses)
        
        # Validation loop if val_loader is provided
        avg_val_loss = None
        if val_loader:
            model.eval()
            val_losses = []
            for images, _ in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                images = images.to(device)
                loss = model.test_step(images, criterion)
                val_losses.append(loss)
            avg_val_loss = sum(val_losses) / len(val_losses)
        
        # Print epoch summary
        if avg_val_loss is not None:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        else:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f}")
        
        # Save checkpoint if checkpoint_dir is specified
        if checkpoint_dir:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.network.state_dict(),
                'ema_model_state_dict': model.ema_network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'normalizer_mean': model.normalizer_mean,
                'normalizer_std': model.normalizer_std
            }
            
            # Save latest checkpoint
            checkpoint_path = os.path.join(checkpoint_dir, f'diffusion_epoch_{epoch+1:03d}.pth')
            torch.save(checkpoint, checkpoint_path)
            
            # Save best model based on validation loss
            if avg_val_loss is not None and avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_checkpoint_path = os.path.join(checkpoint_dir, 'diffusion_best.pth')
                torch.save(checkpoint, best_checkpoint_path)
                print(f"New best model saved at epoch {epoch+1} with val_loss: {avg_val_loss:.4f}")
            
            print(f"Checkpoint saved: {checkpoint_path}")
    
    print("Finished Training Diffusion Model")
    return model


def load_diffusion_checkpoint(model, optimizer, checkpoint_path, device='cpu'):
    """
    Load a saved diffusion model checkpoint and restore model, EMA, and optimizer states.
    
    Args:
        model: DiffusionModel instance
        optimizer: Optimizer instance
        checkpoint_path: Path to checkpoint file
        device: Device to load to
    
    Returns:
        int: Epoch number from checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.network.load_state_dict(checkpoint['model_state_dict'])
    model.ema_network.load_state_dict(checkpoint['ema_model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Restore normalizer settings
    model.normalizer_mean = checkpoint['normalizer_mean']
    model.normalizer_std = checkpoint['normalizer_std']
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Train Loss: {checkpoint['train_loss']:.4f}, Val Loss: {checkpoint['val_loss']}")
    
    return checkpoint['epoch']


def train_ebm(model, data_loader, optimizer, device='cpu', epochs=10, 
              val_loader=None, checkpoint_dir=None):
    """
    Run several iterations of the EBM training loop and return the trained model.
    
    Args:
        model (nn.Module): The EBM model to train.
        data_loader (DataLoader): DataLoader for training data.
        optimizer: Optimizer (e.g., optim.Adam()).
        device (str): Device to train on ('cpu', 'cuda', or 'mps'). Default is 'cpu'.
        epochs (int): Number of training epochs. Default is 10.
        val_loader (DataLoader): Optional DataLoader for validation data.
        checkpoint_dir (str): Optional directory to save checkpoints.
    
    Returns:
        nn.Module: The trained EBM model.
    """
    import os
    from tqdm import tqdm
    
    # Create checkpoint directory if specified
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    model.to(device)
    model.device = device
    
    for epoch in range(epochs):
        model.model.train()
        train_losses = []
        train_metrics = {'loss': 0, 'cdiv_loss': 0, 'reg_loss': 0, 
                        'real_energy': 0, 'fake_energy': 0}
        batch_count = 0
        
        # Training loop with progress bar
        loader_with_progress = tqdm(data_loader, ncols=120, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for images, _ in loader_with_progress:
            images = images.to(device)
            metrics = model.train_step(images, optimizer)
            
            for k, v in metrics.items():
                train_metrics[k] += v
            batch_count += 1
            
            loader_with_progress.set_postfix(loss=f'{metrics["loss"]:.4f}')
        
        # Average training metrics
        for k in train_metrics:
            train_metrics[k] /= batch_count
        
        # Validation loop if val_loader is provided
        val_metrics = None
        if val_loader:
            model.model.eval()
            val_metrics = {'cdiv_loss': 0, 'real_energy': 0, 'fake_energy': 0}
            val_batch_count = 0
            for images, _ in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                images = images.to(device)
                metrics = model.test_step(images)
                for k, v in metrics.items():
                    val_metrics[k] += v
                val_batch_count += 1
            
            for k in val_metrics:
                val_metrics[k] /= val_batch_count
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{epochs} | " + 
              ", ".join(f"{k}: {v:.4f}" for k, v in train_metrics.items()))
        if val_metrics:
            print(f"Validation | " + 
                  ", ".join(f"{k}: {v:.4f}" for k, v in val_metrics.items()))
        
        # Save checkpoint if checkpoint_dir is specified
        if checkpoint_dir:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_metrics['loss'],
                'alpha': model.alpha,
                'steps': model.steps,
                'step_size': model.step_size,
                'noise': model.noise
            }
            
            checkpoint_path = os.path.join(checkpoint_dir, f'ebm_epoch_{epoch+1:03d}.pth')
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
    
    print("Finished Training EBM")
    return model


def load_ebm_checkpoint(model, optimizer, checkpoint_path, device='cpu'):
    """
    Load a saved EBM checkpoint and restore model and optimizer states.
    
    Args:
        model: EBM instance
        optimizer: Optimizer instance
        checkpoint_path: Path to checkpoint file
        device: Device to load to
    
    Returns:
        int: Epoch number from checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Restore hyperparameters
    model.alpha = checkpoint.get('alpha', model.alpha)
    model.steps = checkpoint.get('steps', model.steps)
    model.step_size = checkpoint.get('step_size', model.step_size)
    model.noise = checkpoint.get('noise', model.noise)
    
    print(f"Loaded EBM checkpoint from epoch {checkpoint['epoch']}")
    print(f"Train Loss: {checkpoint['train_loss']:.4f}")
    
    return checkpoint['epoch']

