import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import sys

def create_cost_volume(left_img, right_img, max_disparity=3):
    B, C, H, W = left_img.size()
    print (" B,C,H, W, " , B, C, H, W )
    cost_volume = torch.zeros(B, max_disparity, C * 2, H, W).to(left_img.device)
    
    for d in range(max_disparity):
        if d > 0:
            # Shift the right image by `d` pixels
            shifted_right = F.pad(right_img, (d, 0, 0, 0), mode='constant', value=0)[:, :, :, :-d]
        else:
            shifted_right = right_img
        
        # Concatenate the left image with the shifted right image
        cost_volume[:, d, :, :, :] = torch.cat((left_img, shifted_right), dim=1)
    
    return cost_volume

def compute_disparity_map(cost_volume):
    B, D, C, H, W = cost_volume.size()
    
    # Compute matching cost (L1 distance between concatenated features)
    matching_cost = torch.abs(cost_volume[:, :, :C//2, :, :] - cost_volume[:, :, C//2:, :, :])
    
    # Sum over the channel dimension to get the cost per disparity level
    cost_volume_sum = matching_cost.sum(dim=2)
    
    # Apply softmax to get probability distribution over disparities
    prob_volume = F.softmax(-cost_volume_sum, dim=1)
    
    # Compute disparity map as weighted sum of disparities
    disparity_values = torch.arange(0, D).view(1, D, 1, 1).to(cost_volume.device)
    disparity_map = torch.sum(prob_volume * disparity_values, dim=1)
    
    return disparity_map

def load_image(filepath):
    img = Image.open(filepath).convert('RGB')
    img = np.array(img).astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # Convert to (B, C, H, W) format
    return img

def display_images(left_img, right_img, disparity_map):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Display left image
    axs[0].imshow(left_img.squeeze(0).permute(1, 2, 0))
    axs[0].set_title('Left Image')
    
    # Display right image
    axs[1].imshow(right_img.squeeze(0).permute(1, 2, 0))
    axs[1].set_title('Right Image')
    
    # Display disparity map
    axs[2].imshow(disparity_map.squeeze(0).cpu().detach().numpy(), cmap='inferno')
    axs[2].set_title('Disparity Map')
    
    plt.show()

def main():
    if len(sys.argv) != 3:
        print("Usage: python script_name.py <left_image_path> <right_image_path>")
        return
    
    left_image_path = sys.argv[1]
    right_image_path = sys.argv[2]
    
    # Load images
    left_img = load_image(left_image_path)
    right_img = load_image(right_image_path)
    
    # Create cost volume
    cost_volume = create_cost_volume(left_img, right_img, max_disparity=25)
    
    # Compute disparity map
    disparity_map = compute_disparity_map(cost_volume)
    
    # Display the images and the disparity map
    display_images(left_img, right_img, disparity_map)

if __name__ == "__main__":
    main()

