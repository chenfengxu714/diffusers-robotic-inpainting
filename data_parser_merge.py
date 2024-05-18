import h5py
import os
import cv2
import numpy as np

def copy_h5_structure(source, destination):
    """ Recursively copy the structure of the HDF5 file from source to destination. """
    for key in source.keys():
        item = source[key]
        if isinstance(item, h5py.Dataset):
            destination.create_dataset(key, data=item[()], compression="gzip")
        elif isinstance(item, h5py.Group):
            dest_group = destination.create_group(key)
            copy_h5_structure(item, dest_group)

def write_images_to_h5(file_path, image_folder, output_folder, key="varied_camera_1_left_image_transferred"):
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over all HDF5 files in the directory
    for file in os.listdir(file_path):
        file_dir = os.path.join(file_path, file)
        if not os.path.isdir(file_dir):
            continue

        for fi in os.listdir(file_dir):
            h5_file_path = os.path.join(file_dir, fi)
            if not h5_file_path.endswith('.h5'):
                continue

            image_subfolder = os.path.join(image_folder, file, "final_images")
            if not os.path.exists(image_subfolder):
                print(f"Image subfolder {image_subfolder} does not exist.")
                continue
            
            with h5py.File(h5_file_path, 'r') as h5_file:
                # Check if the hierarchy exists
                if 'observation' in h5_file and 'camera' in h5_file['observation'] and 'image' in h5_file['observation']['camera']:
                    image_group = h5_file['observation']['camera']['image']
                else:
                    print(f"Required hierarchy not found in {h5_file_path}.")
                    continue

                # Read images from the subfolder
                images = []
                for img_name in sorted(os.listdir(image_subfolder), key=lambda x: int(os.path.splitext(x)[0])):
                    img_path = os.path.join(image_subfolder, img_name)
                    img = cv2.imread(img_path)
                    if img is not None:
                        images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                
                if images:
                    images_array = np.array(images)
                    output_h5_dir = os.path.join(output_folder, file)
                    os.makedirs(output_h5_dir, exist_ok=True)
                    output_h5_path = os.path.join(output_h5_dir, fi)

                    # Write to new HDF5 file in the output folder
                    with h5py.File(output_h5_path, 'w') as new_h5_file:
                        # Copy the original structure and datasets
                        copy_h5_structure(h5_file, new_h5_file)
                        
                        # Create the new dataset for the images
                        new_h5_file['observation']['camera']['image'].create_dataset(key, data=images_array, compression="gzip")
                        print(f"Written images to {output_h5_path} under key {key}.")
                else:
                    print(f"No images found in {image_subfolder}.")

# Example usage
file_path = "/rscratch/cfxu/diffusion-RL/style-transfer/data/robo-data/2024-05-10-cup-franka-gripper"
image_folder = "/rscratch/cfxu/diffusion-RL/style-transfer/data/parsered_images_robo/2024-05-10-cup-franka-gripper_masks"
output_folder = "/rscratch/cfxu/diffusion-RL/style-transfer/data/output_h5_2024-05-10-cup-franka-gripper"

write_images_to_h5(file_path, image_folder, output_folder)
