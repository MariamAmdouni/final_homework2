import logging
import os
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import StepLR
from torchvision import models
import torch.nn as nn
from utils import plot_loss_curves


def train_classifier(train_loader, val_loader, criterion, optimizer, num_epochs, model_dir, plot_dir, device,
                     backbone, freeze_backbone, num_classes):
    """
    Trains a CNN for classification with support for fine-tuning

    Parameters:
    -----------
    train_loader : DataLoader
        DataLoader for the training dataset.
    val_loader : DataLoader
        DataLoader for the validation dataset.
    criterion : loss function
        The loss function used for training.
    optimizer : optimizer
        Optimizer for updating model parameters.
    num_epochs : int
        Number of epochs to train the model.
    model_dir : str
        Directory to save the trained model.
    plot_dir : str
        Directory to save training/validation loss plots.
    device : torch.device
        Device to train the model on (e.g., 'cpu' or 'cuda').
    backbone : str
        Name of the model's backbone architecture (e.g., 'resnet18').
    freeze_backbone : bool
        Whether to freeze the backbone layers during training.
    num_classes : int
        Number of output classes for the classification task.
    """

    # Load the pre-trained model
    if backbone == "resnet18":
        model = models.resnet18(pretrained=True)
    elif backbone == "resnet50":
        model = models.resnet50(pretrained=True)
    else:
        raise ValueError("Unsupported backbone model.")

    # Modify the final layer to match the number of classes in the dataset
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Freeze backbone layers if specified
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True  # Only fine-tune the fully connected layer

    model.to(device)  # Move model to the device

    # Ensure the model directory exists
    best_val_loss = float('inf')
    counter = 0
    patience = 10
    train_losses = []
    val_losses = []
    scaler = GradScaler()

    # Learning rate schedule
    scheduler = StepLR(optimizer, step_size=20, gamma=0.1)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass and compute loss inside autocast
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels.long())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()

        # Update learning rate
        scheduler.step()
        average_train_loss = total_train_loss / len(train_loader)
        train_losses.append(average_train_loss)

        # Validation phase
        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                val_outputs = model(images)
                val_loss = criterion(val_outputs, labels.long())
                total_val_loss += val_loss.item()

        average_val_loss = total_val_loss / len(val_loader)
        val_losses.append(average_val_loss)

        logging.info(
            f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {average_train_loss:.8f}, Validation Loss: {average_val_loss:.8f}")

        # Early stopping and model saving
        if average_val_loss < best_val_loss:
            logging.info(f'Validation loss decreased, saved the model at epoch {epoch + 1}')
            best_val_loss = average_val_loss
            counter = 0
            # Save the best trained model
            filename = f'cnn_{backbone}_freeze_backbone_{freeze_backbone}'
            torch.save(model.state_dict(), os.path.join(model_dir, f"{filename}.pth"))
        else:
            counter += 1
            if counter >= patience:
                logging.info(f'Validation loss did not improve for the last {patience} epochs. Stopping early.')
                break

    # Plot loss curves
    plot_loss_curves(train_losses, val_losses, filename, plot_dir)

