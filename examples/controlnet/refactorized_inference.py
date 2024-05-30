import os
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image

class ImageProcessor:
    def __init__(self, base_model_path, controlnet_path, base_image_path, output_base_path, batch_size=128):
        self.base_model_path = base_model_path
        self.controlnet_path = controlnet_path
        self.base_image_path = base_image_path
        self.output_base_path = output_base_path
        self.batch_size = batch_size
        self.prompt = "create a high quality image with a franka robot and white background"
        self.to_tensor = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()
        
        self.controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
        self.pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            base_model_path, controlnet=self.controlnet, torch_dtype=torch.float16
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.safety_checker = None
        self.pipe.requires_safety_checker = False
        self.pipe.enable_sequential_cpu_offload()

    def load_images(self, image_paths):
        images = [load_image(img_path) for img_path in image_paths]
        return images

    def process_folders(self, image_trajectory, image_path): # image_trajectory is M x H x W x 3
        # list_of_folders = [f for f in os.listdir(self.base_image_path) if os.path.isdir(os.path.join(self.base_image_path, f))]
        
        # for folder_name in list_of_folders:
        #     image_path = os.path.join(self.base_image_path, folder_name, 'r2r_images')
        output_folder = 'r2r_transferred'
        output_folder_path = os.path.join(image_path, output_folder)

        if os.path.isdir(output_folder_path):
            print(f"Output folder {output_folder_path} already exists. Skipping...")
            return None
        
        os.makedirs(output_folder_path, exist_ok=True)
        list_of_input_images = [image_trajectory for i in range(image_trajectory.shape[0])]
        outputs = []
        for i in range(0, len(list_of_input_images), self.batch_size):
            batch_images = list_of_input_images[i:i+self.batch_size]    
            prompts = [self.prompt] * len(batch_images)
            generator = torch.manual_seed(1)
            generated_images = self.pipe(
                    prompt=prompts, 
                    num_inference_steps=50, 
                    generator=generator, 
                    image=images, 
                    control_image=images
            ).images
            generated_images = [self.to_pil(self.to_tensor(generated_image).clamp(0, 1)) for generated_image in generated_images]

            for input_image, generated_image in zip(batch_images, generated_images):
                combined_image = [input_image, generated_image]
                widths, heights = zip(*(img.size for img in combined_image))
                total_width = sum(widths)
                max_height = max(heights)
                new_im = Image.new("RGB", (total_width, max_height))
                    
                x_offset = 0
                for im in combined_image:
                    new_im.paste(im, (x_offset, 0))
                    x_offset += im.size[0]
                outputs.append(new_im)
                output_path = os.path.join(output_folder_path, input_image)
                new_im.save(output_path)
                print(f"Saved {output_path}")
                
        return new_im

if __name__ == '__main__':
    base_model_path = "runwayml/stable-diffusion-v1-5"
    controlnet_path = "/rscratch/cfxu/diffusion-RL/style-transfer/data/5-22-franka-ur5/controlnet"
    base_image_path = "/rscratch/cfxu/diffusion-RL/style-transfer/data/parsered_images_robo/2024-05-10-cup-franka-gripper_mask"
    output_base_path = "/rscratch/cfxu/diffusion-RL/style-transfer/data/parsered_images_robo/2024-05-10-tiger-franka-gripper_masks"

    processor = ImageProcessor(base_model_path, controlnet_path, base_image_path, output_base_path, batch_size=128)
    processor.process_folders()
