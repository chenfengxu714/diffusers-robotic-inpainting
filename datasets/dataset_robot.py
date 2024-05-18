import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from einops import repeat
from PIL import Image

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

class Robot_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, low_res):
        # self.transform = transform
        self.low_res = low_res
        self.split = split
        self.image_dir = os.path.join(base_dir, "bowldomain")
        self.mask_dir = os.path.join(base_dir, "bowldomain_mask_only")
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        file_name = self.sample_list[idx].strip('\n')
        
        # Construct paths to the image and its corresponding mask
        image_path = os.path.join(self.image_dir, file_name + '.png')
        mask_path = os.path.join(self.mask_dir, file_name + '_sim_mask.png')
        
        # Load images and masks
        image = Image.open(image_path).convert("RGB")  # Ensure image is RGB
        mask = Image.open(mask_path).convert("L")     # Load mask as grayscale
        # Convert mask to binary 0 and 1
        image = np.array(image)
        mask = np.array(mask)
        mask = (mask > 0).astype(np.uint8)  # Thresholding operation

        sample = {'image': torch.tensor(image), 'label': torch.tensor(mask, dtype=torch.long)}
        label_h, label_w = mask.shape
        low_res_label = zoom(mask, (self.low_res / label_h, self.low_res / label_w), order=0)
        # sample = {'image': image, 'label': label}
        low_res_label = torch.tensor(low_res_label)
        sample = {'image': torch.tensor(image).permute(2,0,1), 'label': torch.tensor(mask, dtype=torch.long), 'low_res_label': low_res_label.long()}
        sample['case_name'] = file_name
        return sample

class RandomGenerator(object):
    def __init__(self, output_size, low_res):
        self.output_size = output_size
        self.low_res = low_res

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        # print(image.shape)
        x, y, c = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        
        image = torch.Tensor(image).permute(2,0,1).contiguous()
        # image = repeat(image, 'c h w -> (repeat c) h w', repeat=3)
        # label = torch.from_numpy(label)
        low_res_label = torch.from_numpy(low_res_label)
        sample = {'image': image, 'label': label, 'low_res_label': low_res_label.long()}
        # print(image.shape, label.shape, low_res_label.shape)
        return sample
        