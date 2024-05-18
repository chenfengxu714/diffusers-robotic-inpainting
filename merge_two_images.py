import os
from PIL import Image
import numpy as np
import cv2
from sklearn.cluster import KMeans
import re

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]


def segment_object(image, num_clusters=5):
    # Reshape the image to a 2D array of pixels
    image = np.array(image)
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # Perform K-means clustering to identify the dominant color
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(pixel_values)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    
    # Reshape the labels to the shape of the original image
    labels = labels.reshape(image.shape[:2])

    # Identify the background label (the most common label)
    unique, counts = np.unique(labels, return_counts=True)
    background_label = unique[np.argmax(counts)]
    
    # Create a mask where the background pixels are marked
    mask = np.zeros(labels.shape, dtype=np.uint8)
    mask[labels == background_label] = 255
    
    # Invert the mask to get the object mask
    object_mask = cv2.bitwise_not(mask)
    
    # Optional: Apply morphological operations to clean up the mask
    # kernel = np.ones((5, 5), np.uint8)
    # object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_CLOSE, kernel)
    # object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_OPEN, kernel)
    
    return object_mask

def paste_objects(background_dir, object_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # List backgrounds and objects
    backgrounds = sorted([f for f in os.listdir(background_dir) if f.endswith(('.png', '.jpg', '.jpeg'))], key=natural_sort_key)
    objects = sorted([f for f in os.listdir(object_dir) if f.endswith(('.png', '.jpg', '.jpeg'))], key=natural_sort_key)

    # Ensure matching counts
    if len(backgrounds) != len(objects):
        print("Error: The number of background images does not match the number of object images.")
        return

    # Loop over paired images
    for bg_filename, obj_filename in zip(backgrounds, objects):
        background_path = os.path.join(background_dir, bg_filename)
        object_path = os.path.join(object_dir, obj_filename)
        
        # Load background image
        background = Image.open(background_path)
        if background is None:
            print(f"Error: Failed to load background image {background_path}")
            continue

        # Load object image
        obj_image = Image.open(object_path)
        if obj_image is None:
            print(f"Error: Failed to load object image {object_path}")
            continue
        
        # Crop the object image to use the right half
        obj_image = obj_image.crop((obj_image.width // 2, 0, obj_image.width, obj_image.height))
        
        # Create a mask from the object image
        mask = segment_object(obj_image)
        
        # Convert mask to a format compatible with PIL (mode 'L' for 8-bit pixels, black and white)
        mask = Image.fromarray(mask).convert("L")
        
        # Paste object using mask at the location (0, 0)
        background.paste(obj_image, (0, 0), mask)

        # Save the composited image
        output_filename = f"{obj_filename}"
        output_path = os.path.join(output_dir, output_filename)
        background.save(output_path)
        print(f"Saved {output_path}")

def process_all_subfolders(base_dir):
    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        if os.path.isdir(folder_path):
            background_dir = os.path.join(folder_path, 'inpainted_frames')
            object_dir = os.path.join(folder_path, 'r2r_transferred')
            output_dir = os.path.join(folder_path, 'final_images')
            if os.path.isdir(background_dir) and os.path.isdir(object_dir):
                paste_objects(background_dir, object_dir, output_dir)
            else:
                print(f"Missing inpainted_frames or r2r_transferred in {folder_path}")

# Example usage
base_dir = '/rscratch/cfxu/diffusion-RL/style-transfer/data/parsered_images_robo/2024-05-11-cup-franka-gripper-background_masks'
process_all_subfolders(base_dir)
