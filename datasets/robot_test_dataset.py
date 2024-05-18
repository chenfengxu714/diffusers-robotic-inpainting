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

class Test_Robot_dataset(Dataset):
    def __init__(self, image_dir):
        # self.transform = transform
        self.path = image_dir
        self.image_dir = os.listdir(image_dir)

    def __len__(self):
        return len(self.image_dir)

    def __getitem__(self, idx):
        # Load images and masks
        image = Image.open(os.path.join(self.path, self.image_dir[idx])).convert("RGB")  # Ensure image is RGB
        image = np.array(image)
        sample = {'image': torch.tensor(image).permute(2,0,1)}
        return sample
