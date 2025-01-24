import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import CustomKeypointDataset
from model import get_pet_nose_model
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import time
import os


def train_model(train_loader, model, criterion, optimizer, num_epochs, device):
    train_losses = []
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        running_loss = 0.0
        total_batches = len(train_loader)
        for batch_idx, (images, heatmaps) in enumerate(train_loader, 1):
            batch_start_time = time.time()

            images, heatmaps = images.to(device), heatmaps.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, heatmaps)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batch_duration = time.time() - batch_start_time
            estimated_time_left = batch_duration * (total_batches - batch_idx)
            print(f"Epoch: {epoch + 1}/{num_epochs}, Batch: {batch_idx}/{total_batches}, Loss: {loss.item():.4f}, Time Left: {estimated_time_left:.2f}s")

        epoch_duration = time.time() - epoch_start_time
        average_loss = running_loss / len(train_loader)
        train_losses.append(average_loss)
        print(f"Epoch {epoch + 1} completed in {epoch_duration:.2f}s, Average Loss: {average_loss:.4f}")

    return model, train_losses

def plot_losses(train_losses, output_dir, model_name="model"):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', marker='o')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"{model_name}_loss_plot.png"))
    plt.close()


parser = argparse.ArgumentParser(description='Training script for pet nose localization.')
parser.add_argument('--data-dir', type=str, required=True, help='Directory with training images')
parser.add_argument('--annotation-file', type=str, required=True, help='Training annotations file')
parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
args = parser.parse_args()

# Dataset and DataLoader setup
train_dataset = CustomKeypointDataset(args.data_dir, args.annotation_file)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

# Model, optimizer, and loss function setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model = get_pet_nose_model().to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.MSELoss()

# Training
trained_model, train_losses = train_model(train_loader, model, criterion, optimizer, args.epochs, device)

output_dir = "output"  # Define your output directory for plots
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
plot_losses(train_losses, output_dir)

# Saving the trained model
torch.save(trained_model.state_dict(), 'trained_model.pth')