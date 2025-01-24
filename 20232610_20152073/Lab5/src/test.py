import argparse
import torch
from dataset import CustomKeypointDataset
from model import get_pet_nose_model
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
import time


def test_model(test_loader, model, device):
    model.eval()
    distances = []
    total_inference_time = 0
    with torch.no_grad():
        for images, true_coords in test_loader:
            images = images.to(device)
            start_time = time.time()
            predicted_coords = model(images).cpu()
            end_time = time.time()
            total_inference_time += (end_time - start_time)

            # Calculate Euclidean distance between predicted and true coordinates
            for pred_coords, gt_coords in zip(predicted_coords, true_coords):
                distance = np.linalg.norm(pred_coords.numpy() - gt_coords.numpy())
                distances.append(distance)
    avg_time_per_image = (total_inference_time / len(test_loader.dataset)) * 1000  # Convert to milliseconds
    return distances, avg_time_per_image


def display_image_with_keypoints(image, keypoints, filename):
    original_size = image.shape[1], image.shape[2]
    keypoints_x = keypoints[0] * original_size[1]  # Width
    keypoints_y = keypoints[1] * original_size[0]  # Height
    plt.imshow(image.permute(1, 2, 0))
    plt.scatter([keypoints_x], [keypoints_y], color='red', marker='o', s=50)
    plt.title('Image with Keypoint')
    plt.savefig(filename)
    plt.close()


parser = argparse.ArgumentParser(description='Testing script for pet nose localization.')
parser.add_argument('--data-dir', type=str, required=True, help='Directory with test images')
parser.add_argument('--annotation-file', type=str, required=True, help='Test annotations file')
parser.add_argument('--model-path', type=str, required=True, help='Path to trained model')
args = parser.parse_args()

# Dataset and DataLoader setup
test_dataset = CustomKeypointDataset(args.data_dir, args.annotation_file)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Model  setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_pet_nose_model().to(device)
model.load_state_dict(torch.load(args.model_path))

distances, avg_time_per_image = test_model(test_loader, model, device)
min_distance = np.min(distances)
mean_distance = np.mean(distances)
max_distance = np.max(distances)
std_distance = np.std(distances)

print(f"Min Distance: {min_distance:.4f}, Mean Distance: {mean_distance:.4f}, Max Distance: {max_distance:.4f}, Std Distance: {std_distance:.4f}")
print(f"Average Inference Time per Image: {avg_time_per_image:.2f} ms")

num_images_to_save = 10
for i in range(num_images_to_save):
    idx = np.random.randint(len(test_dataset))
    image, _ = test_dataset[idx]
    predicted_coords = model(image.unsqueeze(0).to(device)).cpu().detach().squeeze().numpy()
    print("Predicted coordinates:", predicted_coords)
    filename = f"predicted_keypoint_{i}.png"
    display_image_with_keypoints(image, predicted_coords, filename)

