"""
Evaluator Module
Encapsulate evaluation metrics for neural network models.
"""

import torch


def evaluate_model(model, data_loader, criterion, device='cpu'):
    """
    Calculate average loss and accuracy on the test dataset.
    
    Args:
        model (nn.Module): The trained neural network model.
        data_loader (DataLoader): DataLoader for test/validation data.
        criterion: Loss function (e.g., nn.CrossEntropyLoss()).
        device (str): Device to evaluate on ('cpu' or 'cuda'). Default is 'cpu'.
    
    Returns:
        tuple: (avg_loss, accuracy)
            - avg_loss (float): Average loss over all batches.
            - accuracy (float): Accuracy as a percentage (0-100).
    """
    # Move model to the specified device
    model = model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            # Move data to device
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Compute loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            num_batches += 1
            
            # Compute accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # Calculate metrics
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    accuracy = 100 * correct / total if total > 0 else 0.0
    
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    return avg_loss, accuracy

