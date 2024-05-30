import os
import numpy as np
import random
from PIL import Image
import torch
import json
import torchvision.transforms as T
from torchvision.transforms import functional as TF
from scipy.ndimage import zoom
from torch.utils.data import Dataset
from scipy import ndimage
from einops import repeat

class CompositeDataset(Dataset):
    def __init__(self, objects_dir, backgrounds_dir, img_size=(256, 256), low_res=64, transform=None, crop_prob=0.5):
        """
        objects_dir: Directory with subfolders 'images' and 'masks'
        backgrounds_dir: Directory containing background images
        img_size: Final input size of the images (height, width)
        low_res: Low resolution size for labels
        transform: Optional transform to be applied on a sample.
        crop_prob: Probability of applying the crop and resize augmentation.
        """
        with open(objects_dir) as f:
            self.train_data = json.load(f)
        self.objects_images = self.train_data['rgb']
        self.objects_masks = self.train_data['mask']
        if backgrounds_dir == '':
            self.backgrounds = None
        else:
            with open(backgrounds_dir) as ign:
                self.backgrounds = json.load(ign)['rgb']
        self.img_size = img_size
        self.low_res = low_res
        self.transform = transform
        self.crop_prob = crop_prob

    def __len__(self):
        return len(self.objects_images)

    def __getitem__(self, idx):
        # Load object and mask
        obj_image = Image.open(self.objects_images[idx]).convert("RGB")
        obj_image = TF.resize(obj_image, self.img_size)  # Resize to self.img_size
        obj_mask = Image.open(self.objects_masks[idx]).convert("L")
        obj_mask = np.array(obj_mask)
        obj_mask = obj_mask > 128  # Thresholding operation
        obj_mask = TF.to_tensor(obj_mask).long()
        obj_mask = TF.resize(obj_mask, self.img_size, interpolation=T.InterpolationMode.NEAREST)  # Resize to self.img_size

        if self.backgrounds is not None:
            bg_image = Image.open(random.choice(self.backgrounds)).convert("RGB")
            bg_image = TF.resize(bg_image, self.img_size)  # Resize to self.img_size
            bg_image = TF.to_tensor(bg_image)
        
        # Composite object onto background
        obj_image = TF.to_tensor(obj_image)

        if (('rtx' in self.objects_images[idx]) and ('franka_ur5_sawyer_jaco' in self.objects_images[idx])) or self.backgrounds is None:
            composite_image = obj_image
        else:
            # Use the mask to select foreground
            composite_image = obj_image * obj_mask + bg_image * (~obj_mask)

        # Resize the final output to the specified img_size
        composite_image = TF.resize(composite_image, self.img_size, interpolation=T.InterpolationMode.BILINEAR)
        obj_mask = TF.resize(obj_mask, self.img_size, interpolation=T.InterpolationMode.NEAREST)

        _, label_h, label_w = obj_mask.shape
        low_res_label = zoom(obj_mask[0], (self.low_res / label_h, self.low_res / label_w), order=0)
        low_res_label = torch.tensor(low_res_label).long()

        sample = {'image': composite_image, 'label': obj_mask[0], 'low_res_label': low_res_label}
        return sample

    def random_crop_and_resize(self, image, mask):
        # Get bounding box of the mask
        mask_np = mask.squeeze().numpy()
        y_indices, x_indices = np.where(mask_np > 0)
        if len(y_indices) == 0 or len(x_indices) == 0:
            return image, mask

        y_min, y_max = y_indices.min(), y_indices.max()
        x_min, x_max = x_indices.min(), x_indices.max()

        # Define the random crop box
        crop_x_min = max(0, x_min - random.randint(10, 30))
        crop_y_min = max(0, y_min - random.randint(10, 30))
        crop_x_max = min(mask_np.shape[1], x_max + random.randint(10, 30))
        crop_y_max = min(mask_np.shape[0], y_max + random.randint(10, 30))

        # Crop the image and mask
        image = TF.crop(image, crop_y_min, crop_x_min, crop_y_max - crop_y_min, crop_x_max - crop_x_min)
        mask = TF.crop(mask, crop_y_min, crop_x_min, crop_y_max - crop_y_min, crop_x_max - crop_x_min)

        # Resize back to original size
        image = TF.resize(image, (mask_np.shape[0], mask_np.shape[1]), interpolation=T.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, (mask_np.shape[0], mask_np.shape[1]), interpolation=T.InterpolationMode.NEAREST)

        return image, mask

    def additional_augmentations(self, image, mask):
        # Random horizontal flip
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flip
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # Random rotation
        if random.random() > 0.5:
            angle = random.uniform(-30, 30)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)

        # Random color jitter
        if random.random() > 0.5:
            color_jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
            image = color_jitter(image)

        return image, mask


if __name__ == "__main__":
    # Example usage
    from torchvision import transforms
    import matplotlib.pyplot as plt
    dataset = CompositeDataset(
        '/rscratch/cfxu/diffusion-RL/style-transfer/rendering/train.json',
        '/rscratch/cfxu/diffusion-RL/style-transfer/SAMed/datasets/ign_train.json',
        img_size=(256, 256),
        crop_prob=0.5
    )

    sample = dataset[1345]  # Get a sample for demonstration
    composite_image, mask = sample['image'], sample['label']

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
