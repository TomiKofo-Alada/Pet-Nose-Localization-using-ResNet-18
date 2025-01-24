from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import re
import cv2
import numpy as np
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms, models
from torchvision.models.resnet import ResNet18_Weights


class CustomKeypointDataset(Dataset):
    def __init__(self, root_dir, labels_file, target_size=(224, 224)):
        self.root_dir = root_dir
        self.labels_file = labels_file
        self.target_size = target_size
        self.data = self.load_data()

    def load_data(self):
        data = []
        with open(self.labels_file, "r") as file:
            lines = file.readlines()
            for line in lines:
                match = re.match(r'([^,]+),"\((\d+),\s*(\d+)\)"', line.strip())
                if match:
                    image_name, x_str, y_str = match.groups()
                    image_path = os.path.join(self.root_dir, image_name.strip())
                    x, y = int(x_str), int(y_str)
                    data.append((image_path, (x, y)))
                else:
                    print("Invalid line format or unable to parse:", line.strip())
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, keypoints = self.data[idx]
        image = Image.open(image_path).convert('RGB')

        transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image_resized = transform(image)
        image_width, image_height = image.size
        keypoint_normalized = [keypoints[0] / image_width, keypoints[1] / image_height]
        keypoint_normalized = torch.tensor(keypoint_normalized, dtype=torch.float32)
        return image_resized, keypoint_normalized

