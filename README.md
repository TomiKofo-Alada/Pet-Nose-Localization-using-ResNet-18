# Pet-Nose-Localization-using-ResNet-18

## Introduction
This project implements a modified ResNet-18 model to detect and localize pet noses in images. It includes customizations for keypoint detection, adapting the pre-trained ResNet-18 architecture to a regression task for pinpointing (x, y) coordinates of pet noses.

---

## Features
- Modified ResNet-18 architecture for regression tasks.
- Pre-trained weights from PyTorch for feature extraction.
- Adaptive pooling and a custom regression head for coordinate prediction.
- Custom dataset class for keypoint annotation and preprocessing.
- GUI-assisted manual annotation tool using OpenCV for labeling.
- Visualization of predictions on test images with overlayed keypoints.
- Quantitative performance evaluation using Euclidean distance metrics.

---

## Technical Details
- **Framework**: PyTorch
- **Training Platform**: Google Colab with Tesla V100 GPU
- **Hyperparameters**:
  - Batch size: 64
  - Learning rate: 0.001
  - Loss function: Mean Squared Error (MSE)
  - Optimizer: Adam
  - Epochs: 25
- **Average Inference Time**: 2.14 ms/image

---

## Results
- **Mean Euclidean Distance**: 0.0599
- **Standard Deviation**: 0.0464
- **Minimum Distance**: 0.0016
- **Maximum Distance**: 0.3923


---



### Prerequisites
Ensure you have Python 3.8 or higher and the required dependencies installed.


## Model Architecture
- **Base Model**: ResNet-18
- **Customizations**:
  - Retained all layers up to the second-to-last layer for feature extraction.
  - Added an adaptive average pooling layer for consistent output size.
  - Replaced the classification head with a regression head for predicting (x, y) coordinates.
- **Preprocessing**:
  - Resized images to 224x224.
  - Normalized using ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).

---

## Performance Metrics
- The model's performance was evaluated using the Euclidean distance between predicted and actual keypoints:
  - **Mean Distance**: 0.0599
  - **Standard Deviation**: 0.0464
  - **Maximum Distance**: 0.3923

---

## Tools Used
- **Data Annotation**: Custom GUI tool built with OpenCV for manual keypoint labeling.
- **Training**: PyTorch DataLoader for efficient batch processing.
- **Evaluation**: Custom testing script with keypoint visualization and performance metrics.

---

## Acknowledgments
- [ResNet Paper](https://doi.org/10.1109/cvpr.2016.90)
- [Torchvision Documentation](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html)
- [Google Colab](https://colab.research.google.com/)

---


