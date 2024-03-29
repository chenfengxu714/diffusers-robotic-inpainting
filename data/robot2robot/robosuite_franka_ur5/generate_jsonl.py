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
IMAGES_PATH = "/home/lawrence/xembody_followup/robot2robot/rendering/paired_images_0/ur5e_rgb"

# 2. input_image can be the target robot before masking or the analytically inpainted image
INPUT_IMAGES_PATH = "/home/lawrence/xembody_followup/robot2robot/rendering/paired_images_0/panda_rgb"

# 3. mask is the union of the target robot mask and the ground truth robot mask or all 0s
MASKS_PATH = "/home/lawrence/xembody_followup/robot2robot/rendering/paired_images_0/panda_mask"

# 4. conditioning_image can be the target robot before masking or the analytically inpainted image
CONDITIONING_IMAGES_PATH = INPUT_IMAGES_PATH


jsonl_data = []


text = "create a high quality image with a UR5 robot and white background"



for i in range(9):
    images_path = IMAGES_PATH.replace("_0", "_" + str(i))
    input_images_path = INPUT_IMAGES_PATH.replace("_0", "_" + str(i))
    masks_path = MASKS_PATH.replace("_0", "_" + str(i))
    # Iterate through the folders and pair images with the same indices
    for ur5e_subfolder, franka_subfolder in zip(sorted(os.listdir(images_path)), sorted(os.listdir(input_images_path))):
        ur5e_path = os.path.join(images_path, ur5e_subfolder)
        franka_path = os.path.join(input_images_path, franka_subfolder)
        if "dummy" in masks_path:
            mask_path = masks_path
        else:
            mask_path = os.path.join(masks_path, franka_subfolder)

        # Check if both subfolders exist and are directories
        if os.path.isdir(ur5e_path) and os.path.isdir(franka_path):
            ur5e_images = sorted([os.path.join(ur5e_path, img) for img in os.listdir(ur5e_path)])
            franka_images = sorted([os.path.join(franka_path, img) for img in os.listdir(franka_path)])
            if "dummy" in masks_path:
                mask_images = [os.path.join(mask_path, "0.jpg") for _ in range(len(ur5e_images))]
            else:
                mask_images = sorted([os.path.join(mask_path, img) for img in os.listdir(mask_path)])
                

            # Ensure both folders have the same number of images
            if len(ur5e_images) == len(franka_images) and len(ur5e_images) == len(mask_images):
                for ur5e_img, franka_img, mask_img in zip(ur5e_images, franka_images, mask_images):
                    data_entry = {
                        "text": text,
                        "image": ur5e_img, #'/'.join(franka_img.split('/')[-2:]),  # Path to UR5 image
                        "input_image": franka_img, #'/'.join(ur5e_img.split('/')[-2:])  # Path to Franka image
                        # "mask": mask_img, #'/'.join(mask_img.split('/')[-2:])  # Path to mask image
                        # "conditioning_image": franka_img, #'/'.join(ur5e_img.split('/')[-2:])  # Path to UR5 image
                    }
                    jsonl_data.append(data_entry)

            

# Write data to a JSONL file
with open('paired_images_sample.jsonl', 'w') as outfile:
    for entry in jsonl_data:
        json.dump(entry, outfile)
        outfile.write('\n')


