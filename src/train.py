import argparse
import logging
import os

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import StepLR
import mlflow
import mlflow.pytorch

from utils import plot_loss_curves

# Add argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Train a CNN model for classification")
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train the model')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--model_dir', type=str, default='./models', help='Directory to save the trained model')
    parser.add_argument('--plot_dir', type=str, default='./plots', help='Directory to save training/validation loss plots')
    parser.add_argument('--backbone', type=str, default='resnet18', help='Backbone model for the CNN')
    parser.add_argument('--freeze_backbone', type=bool, default=True, help='Whether to freeze the backbone layers during training')
    return parser.parse_args()

def train_classifier_with_mlflow(model, train_loader, val_loader, criterion, optimizer, num_epochs, model_dir, plot_dir, device, backbone, freeze_backbone):
    """
    Trains a CNN for classification and logs the experiment with MLflow

    Parameters:
    -----------
    model : nn.Module
        The model to be trained.
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
        Name of the model's backbone architecture.
    freeze_backbone : bool
        Whether to freeze the backbone layers during training.

    Returns:
    --------
    None
    """
    # Start the MLflow experiment
    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_param("backbone", backbone)
        mlflow.log_param("freeze_backbone", freeze_backbone)
        mlflow.log_param("epochs", num_epochs)
        mlflow.log_param("batch_size", batch_size)

        # Ensure the model directory exists
        global filename
        best_val_loss = float('inf')
        counter = 0
        patience = 10
        train_losses = []
        val_losses = []
        scaler = GradScaler()

        # Learning rate schedule
        scheduler = StepLR(optimizer, step_size=20, gamma=0.1)

        model.to(device)  # Move model to the device

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

            # Log metrics to MLflow
            mlflow.log_metric("train_loss", average_train_loss)
            mlflow.log_metric("val_loss", average_val_loss)

            # Early stopping and model saving
            if average_val_loss < best_val_loss:
                logging.info(f'Validation loss decreased, saved the model at epoch {epoch + 1}')
                best_val_loss = average_val_loss
                counter = 0
                # Save the best trained model
                filename = f'cnn_{backbone}_freeze_backbone_{freeze_backbone}'
                torch.save(model.state_dict(), os.path.join(model_dir, f"{filename}.pth"))
                # Log model artifact to MLflow
                mlflow.pytorch.log_model(model, "model")
            else:
                counter += 1
                if counter >= patience:
                    logging.info(f'Validation loss did not improve for the last {patience} epochs. Stopping early.')
                    break

        # Plot loss curves
        plot_loss_curves(train_losses, val_losses, filename, plot_dir)
        # Log plot artifact to MLflow
        mlflow.log_artifact(os.path.join(plot_dir, f"{filename}_loss_curves.png"))

if __name__ == "__main__":
    # Parse the arguments
    args = parse_args()

    # Example: You need to define the model, optimizer, dataloaders, etc., as per your setup.
    model = YourModel()  # Example model
    train_loader = get_train_loader(args.batch_size)  # Load the train data
    val_loader = get_val_loader(args.batch_size)  # Load the validation data
    criterion = torch.nn.CrossEntropyLoss()  # Define loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)  # Define optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Start training
    train_classifier_with_mlflow(model, train_loader, val_loader, criterion, optimizer, args.epochs, args.model_dir, args.plot_dir, device, args.backbone, args.freeze_backbone)
