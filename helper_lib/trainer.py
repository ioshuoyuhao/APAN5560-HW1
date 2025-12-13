"""
Trainer Module
Abstract the training loop for neural network models.
"""

import torch


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

