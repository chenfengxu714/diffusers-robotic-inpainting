import os
import numpy as np
import random
from PIL import Image
import torch
import json
import torchvision.transforms as T
from torchvision.transforms import functional as TF
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from scipy import ndimage
from einops import repeat
from PIL import Image

class CompositeDataset(Dataset):
    def __init__(self, objects_dir, backgrounds_dir, low_res=64, transform=None):
        """
        objects_dir: Directory with subfolders 'images' and 'masks'
        backgrounds_dir: Directory containing background images
        transform: Optional transform to be applied on a sample.
        """
        f = open(objects_dir)
        self.train_data = json.load(f)
        self.objects_images = self.train_data['rgb']
        self.objects_masks = self.train_data['mask']
        if backgrounds_dir == '':
            self.backgrounds = None
        else:
            ign = open(backgrounds_dir)
            self.backgrounds = json.load(ign)['rgb']
        self.low_res = low_res
        self.transform = transform

    def __len__(self):
        return len(self.objects_images)

    def __getitem__(self, idx):
        # Load object and mask
        obj_image = Image.open(self.objects_images[idx]).convert("RGB")
        
        obj_mask = Image.open(self.objects_masks[idx]).convert("L")
        obj_mask = np.array(obj_mask)
        obj_mask = obj_mask > 128 # Thresholding operation

   
        bg_image = Image.open(random.choice(self.backgrounds)).convert("RGB")

        # Ensure background and object are the same size
        bg_image = TF.resize(bg_image, obj_image.size)

        # Composite object onto background
        obj_image = TF.to_tensor(obj_image)
        bg_image = TF.to_tensor(bg_image)

        if "sim_rollout_images" in self.objects_images[idx]:
            composite_image = obj_image
        else:
            # Use the mask to select foreground
            composite_image = obj_image * obj_mask + bg_image * (~obj_mask)
        # composite_image = obj_image
        # Prepare mask for compositing
        # obj_mask = TF.resize(obj_mask, obj_image.size, interpolation=T.InterpolationMode.NEAREST)
        obj_mask = TF.to_tensor(obj_mask).long()
        _, label_h, label_w = obj_mask.shape
        low_res_label = zoom(obj_mask[0], (self.low_res / label_h, self.low_res / label_w), order=0)
        # sample = {'image': image, 'label': label}
        low_res_label = torch.tensor(low_res_label).long()

        sample = {'image': composite_image, 'label': obj_mask[0], 'low_res_label': low_res_label}
        return sample


        

if __name__ == "__main__":
    # Example usage
    from torchvision import transforms
    import matplotlib.pyplot as plt
    dataset = CompositeDataset('/rscratch/cfxu/diffusion-RL/style-transfer/rendering/train.json', '/rscratch/cfxu/diffusion-RL/style-transfer/SAMed/datasets/ign_train.json')

    composite_image, _, mask = dataset[1345]  # Get the first sample for demonstration
    
    # Convert tensors to numpy arrays for plotting
    composite_image = composite_image.permute(1, 2, 0).numpy()  # Convert CHW to HWC
    mask = mask.numpy()
    # Plotting the images
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(composite_image)
    ax[0].set_title('Composite Image')
    ax[0].axis('off')

    ax[1].imshow(mask, cmap='gray')
    ax[1].set_title('Mask')
    ax[1].axis('off')

    plt.savefig('example.png')