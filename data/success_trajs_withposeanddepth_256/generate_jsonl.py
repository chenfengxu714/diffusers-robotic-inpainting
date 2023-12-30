import os
import json


# For masked inpainting, the 4 images are
# "image" (ground truth image): franka_rgb
# "input_image" (image to be masked): ur5_rgb
# "mask" (mask): union of ur5_mask and franka_mask will be 0s (masked), everything else will be 1s (not masked)
# "conditioning_image" (image to be masked): ur5_rgb

# For analytic inpainting improvement, the 4 images are
# "image" (ground truth image): franka_rgb
# "input_image" (analytically inpainted): franka_analytic_inpainted
# "mask" (mask): all 0s (everything is black, meaning not masked)
# "conditioning_image" (analytically inpainted): franka_analytic_inpainted

# 1. image is always the ground truth image
IMAGES_PATH = "/home/lawrence/diffusers-robotic-inpainting/data/success_trajs_withposeanddepth_256/franka_rgb"

# 2. input_image can be the target robot before masking or the analytically inpainted image
INPUT_IMAGES_PATH = "/home/lawrence/diffusers-robotic-inpainting/data/success_trajs_withposeanddepth_256/ur5e_rgb"
# INPUT_IMAGES_PATH = "/home/lawrence/diffusers-robotic-inpainting/data/success_trajs_withposeanddepth_256/franka_analytic_inpainted"

# 3. mask is the union of the target robot mask and the ground truth robot mask or all 0s
MASKS_PATH = "/home/lawrence/diffusers-robotic-inpainting/data/success_trajs_withposeanddepth_256/union_mask"
# MASKS_PATH = "/home/lawrence/diffusers-robotic-inpainting/data/success_trajs_withposeanddepth_256/dummy_mask"

# 4. conditioning_image can be the target robot before masking or the analytically inpainted image
CONDITIONING_IMAGES_PATH = "/home/lawrence/diffusers-robotic-inpainting/data/success_trajs_withposeanddepth_256/ur5e_rgb"
# CONDITIONING_IMAGES_PATH = "/home/lawrence/diffusers-robotic-inpainting/data/success_trajs_withposeanddepth_256/masked_images"
# CONDITIONING_IMAGES_PATH = "/home/lawrence/diffusers-robotic-inpainting/data/success_trajs_withposeanddepth_256/franka_analytic_inpainted"


jsonl_data = []

# text = "replace the UR5 robot and its gripper with a Franka Panda robot and gripper at the exact same position and orientation"
# text = "inpaint the masked region with a Franka Panda robot"
# text = "remove artifacts in the image"
text = "create a high quality image with a Franka Panda robot, a table, and a red cube on the table"




# Iterate through the folders and pair images with the same indices
for ur5e_subfolder, franka_subfolder in zip(sorted(os.listdir(INPUT_IMAGES_PATH)), sorted(os.listdir(IMAGES_PATH))):
    ur5e_path = os.path.join(INPUT_IMAGES_PATH, ur5e_subfolder)
    franka_path = os.path.join(IMAGES_PATH, franka_subfolder)
    if "dummy" in MASKS_PATH:
        mask_path = MASKS_PATH
    else:
        mask_path = os.path.join(MASKS_PATH, franka_subfolder)

    # Check if both subfolders exist and are directories
    if os.path.isdir(ur5e_path) and os.path.isdir(franka_path):
        ur5e_images = sorted([os.path.join(ur5e_path, img) for img in os.listdir(ur5e_path)])
        franka_images = sorted([os.path.join(franka_path, img) for img in os.listdir(franka_path)])
        if "dummy" in MASKS_PATH:
            mask_images = [os.path.join(mask_path, "0.jpg") for _ in range(len(ur5e_images))]
        else:
            mask_images = sorted([os.path.join(mask_path, img) for img in os.listdir(mask_path)])
            

        # Ensure both folders have the same number of images
        if len(ur5e_images) == len(franka_images) and len(ur5e_images) == len(mask_images):
            for ur5e_img, franka_img, mask_img in zip(ur5e_images, franka_images, mask_images):
                data_entry = {
                    "text": text,
                    "image": franka_img, #'/'.join(franka_img.split('/')[-2:]),  # Path to Franka Panda image
                    "input_image": ur5e_img, #'/'.join(ur5e_img.split('/')[-2:])  # Path to UR5 image
                    "mask": mask_img, #'/'.join(mask_img.split('/')[-2:])  # Path to mask image
                    "conditioning_image": ur5e_img, #'/'.join(ur5e_img.split('/')[-2:])  # Path to UR5 image
                }
                jsonl_data.append(data_entry)
        else:
            print(f"Folder {ur5e_subfolder} has {len(ur5e_images)} UR5 images and {len(franka_images)} Franka images")
            for ur5e_img in ur5e_images:
                data_entry = {
                    "text": text,
                    "image": ur5e_img.replace('franka_analytic_inpainted', 'franka_rgb'), #'/'.join(ur5e_img.split('/')[-2:]),  # Path to Franka Panda image
                    "input_image": ur5e_img, #'/'.join(ur5e_img.split('/')[-2:])  # Path to UR5 image
                    "mask": os.path.join(mask_path, "0.jpg"),  # Path to mask image
                    "conditioning_image": ur5e_img, #'/'.join(ur5e_img.split('/')[-2:])  # Path to UR5 image
                }
                jsonl_data.append(data_entry)

# Write data to a JSONL file
with open('paired_images_mask_inpaint_new.jsonl', 'w') as outfile:
    for entry in jsonl_data:
        json.dump(entry, outfile)
        outfile.write('\n')


