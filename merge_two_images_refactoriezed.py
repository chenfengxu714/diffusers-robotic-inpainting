import os
from PIL import Image
import numpy as np
import cv2
import re
import random

class ImageProcessor:
    def __init__(self, base_dir):
        self.base_dir = base_dir

    def change_brightness(self, img, mean_value=100, mask=None, randomness=0):
        value = mean_value + random.randint(-randomness, randomness)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        if mask is None:
            mask = np.ones_like(v)
        else:
            mask = mask.squeeze()
        
        if value > 0:
            lim = 255 - value
            v[(v > lim) & (mask == 1)] = 255
            v[(v <= lim) & (mask == 1)] += value
        else:
            lim = -value
            v[(v < lim) & (mask == 1)] = 0
            v[(v >= lim) & (mask == 1)] -= lim

        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img

    @staticmethod
    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

    def segment_object(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_bound = np.array([0, 0, 200])
        upper_bound = np.array([180, 50, 255])
        
        color_mask = cv2.inRange(hsv, lower_bound, upper_bound)
        object_mask = cv2.bitwise_not(color_mask)
        
        edges = cv2.Canny(image, 100, 200)
        combined_mask = cv2.bitwise_or(object_mask, edges)
        
        kernel = np.ones((3, 3), np.uint8)
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        eroded_mask = cv2.erode(combined_mask, kernel=np.ones((5, 5), np.uint8), iterations=1)
        final_mask = cv2.GaussianBlur(eroded_mask, (3, 3), 0)
        
        return final_mask

    def paste_objects(self, backgrounds, objects, output_dir_base, randomness=0):
        output_dir = f"{output_dir_base}_{randomness}"
        os.makedirs(output_dir, exist_ok=True)
        
        if len(backgrounds) != len(objects):
            print("Error: The number of background images does not match the number of object images.")
            return

        for i, (background, obj_image) in enumerate(zip(backgrounds, objects)):
            obj_image = Image.fromarray(obj_image)
            background = Image.fromarray(background)
            
            obj_image = obj_image.crop((obj_image.width // 2, 0, obj_image.width, obj_image.height))
            mask = self.segment_object(np.array(obj_image))
            brightened_obj_image = self.change_brightness(np.array(obj_image), mean_value=50, randomness=randomness)
            mask = Image.fromarray(mask).convert("L")
            obj_image = Image.fromarray(brightened_obj_image).convert('RGBA')

            if mask.size != obj_image.size:
                mask = mask.resize(obj_image.size, Image.ANTIALIAS)
            
            background.paste(obj_image, (0, 0), mask)
            output_filename = f"output_{i:04d}.png"
            output_path = os.path.join(output_dir, output_filename)
            background.save(output_path)
            print(f"Saved {output_path}")

    def process_all_subfolders(self):
        for folder_name in os.listdir(self.base_dir):
            folder_path = os.path.join(self.base_dir, folder_name)
            if os.path.isdir(folder_path):
                background_dir = os.path.join(folder_path, 'inpainted_frames')
                object_dir = os.path.join(folder_path, 'r2r_transferred')
                output_dir_base = os.path.join(folder_path, 'final_images_brighted')
                if os.path.isdir(background_dir) and os.path.isdir(object_dir):
                    backgrounds = [np.array(Image.open(os.path.join(background_dir, f))) for f in sorted(os.listdir(background_dir), key=self.natural_sort_key) if f.endswith(('.png', '.jpg', '.jpeg'))]
                    objects = [np.array(Image.open(os.path.join(object_dir, f))) for f in sorted(os.listdir(object_dir), key=self.natural_sort_key) if f.endswith(('.png', '.jpg', '.jpeg'))]
                    self.paste_objects(backgrounds, objects, output_dir_base, randomness=0)
                    self.paste_objects(backgrounds, objects, output_dir_base, randomness=30)
                else:
                    print(f"Missing inpainted_frames or r2r_transferred in {folder_path}")

# Example usage
base_dir = '/rscratch/cfxu/diffusion-RL/style-transfer/data/parsered_images_robo/2024-05-10-cup-franka-gripper_mask'
processor = ImageProcessor(base_dir)
processor.process_all_subfolders()
