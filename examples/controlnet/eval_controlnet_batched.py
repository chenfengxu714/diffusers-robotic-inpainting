from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import torch
from torchvision import transforms
import numpy as np
import os
from PIL import Image

# Paths and models
base_model_path = "runwayml/stable-diffusion-v1-5"
controlnet_path = "/rscratch/cfxu/diffusion-RL/style-transfer/data/controlnet"
base_image_path = "/rscratch/cfxu/diffusion-RL/style-transfer/data/parsered_images_robo/2024-05-11-cup-franka-gripper-background_masks"
# output_base_path = "/rscratch/cfxu/diffusion-RL/style-transfer/data/parsered_images_robo/2024-05-10-tiger-franka-gripper_masks"

# Load ControlNet and pipeline
controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    base_model_path, controlnet=controlnet, torch_dtype=torch.float16
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.safety_checker = None
pipe.requires_safety_checker = False
pipe.enable_sequential_cpu_offload()

# List all folders in the base image directory
list_of_folders = [f for f in os.listdir(base_image_path) if os.path.isdir(os.path.join(base_image_path, f))]

# Helper function to load and process images
def load_images(image_paths):
    images = [load_image(img_path) for img_path in image_paths]
    return images

# Batch process images
batch_size = 128  # Adjust batch size as needed
prompt = "create a high quality image with a ur5 robot and white background"

for folder_name in list_of_folders:
    image_path = os.path.join(base_image_path, folder_name, 'r2r_images')
    output_folder = f"r2r_transferred"
    output_folder_path = os.path.join(base_image_path, folder_name, output_folder)

    # Skip if the output folder already exists
    if os.path.isdir(output_folder_path):
        print(f"Output folder {output_folder_path} already exists. Skipping...")
        continue

    # Create the output folder
    os.makedirs(output_folder_path, exist_ok=True)

    list_of_input_images = sorted(os.listdir(image_path))
    
    for i in range(0, len(list_of_input_images), batch_size):
        batch_images = list_of_input_images[i:i+batch_size]
        images = load_images([os.path.join(image_path, img) for img in batch_images])
        
        # Generate images in batch
        prompts = [prompt] * len(images)
        generator = torch.manual_seed(1)
        generated_images = pipe(
            prompt=prompts, 
            num_inference_steps=50, 
            generator=generator, 
            image=images, 
            control_image=images
        ).images
        
        # Save the generated images
        for input_image, generated_image in zip(batch_images, generated_images):
            original_image = load_image(os.path.join(image_path, input_image))
            combined_image = [original_image, generated_image]
            
            # Create a new image combining original and generated
            widths, heights = zip(*(img.size for img in combined_image))
            total_width = sum(widths)
            max_height = max(heights)
            new_im = Image.new("RGB", (total_width, max_height))
            
            x_offset = 0
            for im in combined_image:
                new_im.paste(im, (x_offset, 0))
                x_offset += im.size[0]
            
            output_path = os.path.join(output_folder_path, input_image)
            new_im.save(output_path)
            print(f"Saved {output_path}")
