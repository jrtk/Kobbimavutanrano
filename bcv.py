import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image  # Ensure this line is present to import Image from PIL
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Example Convolutional Neural Network for feature extraction
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x

# Function to prepare the image for the network
def prepare_image(image_path, target_size=(128, 128)):
    # Step 1: Load the image using PIL
    image = Image.open(image_path).convert('RGB')  # Convert to RGB if it's grayscale

    # Step 2: Define the transformations (resize, convert to tensor, normalize)
    transform = transforms.Compose([
        transforms.Resize(target_size),  # Resize to target size
        transforms.ToTensor(),           # Convert the image to a PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])

    # Step 3: Apply the transformations
    image_tensor = transform(image)  # Now, this is a tensor with shape [C, H, W]

    # Step 4: Add batch dimension (PyTorch expects input as [B, C, H, W])
    image_tensor = image_tensor.unsqueeze(0)  # Shape: [1, C, H, W]

    return image_tensor

# Function to construct the 3D cost volume
def construct_cost_volume(left_feature, right_feature, max_disparity):
    batch_size, channels, height, width = left_feature.size()
    cost_volume = torch.zeros(batch_size, channels * 2, max_disparity, height, width).to(left_feature.device)
    
    for d in range(max_disparity):
        if d > 0:
            cost_volume[:, :channels, d, :, d:] = left_feature[:, :, :, d:]
            cost_volume[:, channels:, d, :, d:] = right_feature[:, :, :, :-d]
        else:
            cost_volume[:, :channels, d, :, :] = left_feature
            cost_volume[:, channels:, d, :, :] = right_feature

    return cost_volume

# Visualize a specific slice of the cost volume for a given disparity level
def visualize_cost_slice(cost_volume, disparity_level):
    # Extract the cost slice for the specified disparity level
    cost_slice = cost_volume[0, :, disparity_level, :, :].cpu().detach().numpy()
    
    # Sum the channels to get a single 2D map
    cost_map = np.sum(cost_slice, axis=0)
    
    plt.figure(figsize=(10, 8))
    plt.title(f"Cost Volume Slice at Disparity Level {disparity_level}")
    plt.imshow(cost_map, cmap='viridis')
    plt.colorbar()
    plt.show()

# Visualize the minimum cost map across all disparity levels
def visualize_min_cost_map(cost_volume):
    # Calculate the minimum cost for each pixel across all disparity levels
    min_cost_map = torch.min(cost_volume, dim=2)[0].cpu().detach().numpy()
    
    # Sum the channels to get a single 2D map
    min_cost_map = np.sum(min_cost_map[0], axis=0)
    
    plt.figure(figsize=(10, 8))
    plt.title("Minimum Cost Map")
    plt.imshow(min_cost_map, cmap='viridis')
    plt.colorbar()
    plt.show()

# Simulating inference on a stereo image pair
left_image_path = 'im0.png'  # Replace with your actual left image path
right_image_path = 'im1.png'  # Replace with your actual right image path

# Load and prepare the images
left_image_tensor = prepare_image(left_image_path)
right_image_tensor = prepare_image(right_image_path)

# Initialize the feature extractor
feature_extractor = FeatureExtractor()

# Feature extraction
left_feature = feature_extractor(left_image_tensor)
right_feature = feature_extractor(right_image_tensor)

# Set the maximum disparity (e.g., 64)
max_disparity = 64

# Constructing the cost volume
cost_volume = construct_cost_volume(left_feature, right_feature, max_disparity)

# Visualize the cost slice for disparity level 32 (middle disparity)
visualize_cost_slice(cost_volume, disparity_level=4)

# Visualize the minimum cost map across all disparity levels
visualize_min_cost_map(cost_volume)

