import os
import random

def create_split(base_dir, split_ratio=0.2):
    image_dir = os.path.join(base_dir, 'bowldomain')
    files = os.listdir(image_dir)
    files = [f for f in files if f.endswith('.png')]  # Ensure only PNG files are listed

    # Randomly shuffle the list of files
    random.shuffle(files)
    
    # Calculate the split index
    num_files = len(files)
    split_index = int(num_files * split_ratio)
    
    # Split the files into training and validation sets
    validation_files = files[:split_index]
    train_files = files[split_index:]
    
    # Write the filenames to their respective files
    with open(os.path.join(base_dir, 'train.txt'), 'w') as f:
        for file in train_files:
            # Write only the basename without extension
            f.write(os.path.splitext(file)[0] + '\n')

    with open(os.path.join(base_dir, 'val.txt'), 'w') as f:
        for file in validation_files:
            # Write only the basename without extension
            f.write(os.path.splitext(file)[0] + '\n')

# Usage
base_dir = '/rscratch/cfxu/diffusion-RL/style-transfer/diffusers-robotic-inpainting/data'  # Update this to your actual directory
create_split(base_dir, split_ratio=0.2)  # Approximately 1:5 ratio
