# https://github.com/huggingface/diffusers/tree/main/examples/controlnet
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler, StableDiffusionControlNetImg2ImgPipeline
from diffusers.utils import load_image
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
base_model_path = "runwayml/stable-diffusion-v1-5"
# controlnet_path = "/shared/projects/diffusers/outputs/ur5_to_franka_diverse_angles/5e-5/checkpoint-5000/controlnet"
# controlnet_path = "/shared/projects/diffusers/outputs/ur5_to_franka_diverse_angles/1e-4/checkpoint-5000/controlnet"
controlnet_path = "/shared/projects/diffusers/outputs/franka_to_ur5_diverse_angles/lr_1e-4_bs_512/checkpoint-11350/controlnet"
controlnet_path = "/shared/projects/diffusers/outputs/franka_to_ur5_diverse_angles_viola_finetune/lr_5e-5_bs_512/checkpoint-1000/controlnet"
controlnet_path = "/shared/projects/diffusers/outputs/franka_to_ur5_diverse_angles_mirage_finetune/lr_5e-5_bs_512/checkpoint-300/controlnet"
controlnet_path = "/shared/projects/diffusers/outputs/franka_to_ur5_diverse_angles_all/lr_1e-4_bs_512/checkpoint-2150/controlnet"


controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
# pipe = StableDiffusionControlNetPipeline.from_pretrained(
#     base_model_path, controlnet=controlnet, torch_dtype=torch.float16
# )
pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    base_model_path, controlnet=controlnet, torch_dtype=torch.float16
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.safety_checker = None
pipe.requires_safety_checker = False
# speed up diffusion process with faster scheduler and memory optimization
# pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# remove following line if xformers is not installed or when using Torch 2.0.
# pipe.enable_xformers_memory_efficient_attention()
# memory optimization.
# pipe.enable_model_cpu_offload()
pipe.enable_sequential_cpu_offload()


list_of_input_images = ["/home/lawrence/xembody_followup/robot2robot/rendering/paired_images_0/panda_rgb/0/10.jpg",
                        "/home/lawrence/xembody_followup/robot2robot/rendering/paired_images_1/panda_rgb/0/10.jpg",
                        "/home/lawrence/xembody_followup/robot2robot/rendering/paired_images_2/panda_rgb/0/10.jpg",
                        "/home/lawrence/xembody_followup/robot2robot/rendering/paired_images_3/panda_rgb/0/10.jpg",
                        "/home/lawrence/xembody_followup/robot2robot/rendering/paired_images_4/panda_rgb/0/10.jpg",
                        "/home/lawrence/xembody_followup/robot2robot/rendering/paired_images_5/panda_rgb/0/10.jpg",
                        "/home/lawrence/xembody_followup/robot2robot/rendering/paired_images_6/panda_rgb/0/10.jpg",
                        "/home/lawrence/xembody_followup/robot2robot/rendering/paired_images_7/panda_rgb/0/10.jpg",
                        "/home/lawrence/xembody_followup/robot2robot/rendering/paired_images_8/panda_rgb/2309/10.jpg"]

list_of_input_images = ["/home/lawrence/xembody_followup/viola_dataset/plateforkdomain/60_sim.png",
                        "/home/lawrence/xembody_followup/viola_dataset/plateforkdomain/140_sim.png",
                        "/home/lawrence/xembody_followup/viola_dataset/plateforkdomain/80_sim.png"]

# list_of_input_images = ["/home/lawrence/xembody_followup/robot2robot/rendering/paired_images_1/ur5e_rgb/4898/10.jpg",
#                         "/home/lawrence/xembody_followup/robot2robot/rendering/paired_images_2/ur5e_rgb/4898/10.jpg",
#                         "/home/lawrence/xembody_followup/robot2robot/rendering/paired_images_3/ur5e_rgb/4898/10.jpg",
#                         "/home/lawrence/xembody_followup/robot2robot/rendering/paired_images_4/ur5e_rgb/4898/10.jpg",
#                         "/home/lawrence/xembody_followup/robot2robot/rendering/paired_images_5/ur5e_rgb/4898/10.jpg",
#                         "/home/lawrence/xembody_followup/robot2robot/rendering/paired_images_6/ur5e_rgb/4898/10.jpg",
#                         "/home/lawrence/xembody_followup/robot2robot/rendering/paired_images_7/ur5e_rgb/4898/10.jpg",
#                         "/home/lawrence/xembody_followup/robot2robot/rendering/paired_images_8/ur5e_rgb/4898/10.jpg"]

list_of_input_images = ["/home/lawrence/xembody_followup/mirage_data/test0_cropped_resized.png",
                        "/home/lawrence/xembody_followup/mirage_data/test1_cropped_resized.png",
                        "/home/lawrence/xembody_followup/mirage_data/test2_cropped_resized.png",
                        "/home/lawrence/xembody_followup/mirage_data/test3_cropped_resized.png"]

for i, input_image in enumerate(list_of_input_images):
    image = load_image(input_image)
    prompt = "create a high quality image with a UR5 robot and white background"
    # generate image
    generator = torch.manual_seed(1)
    generated_image = pipe(
        prompt, num_inference_steps=20, generator=generator, image=image, control_image=image
    ).images[0]
    # ground_truth_image = load_image(input_image.replace("ur5e", "panda"))
    # save an image of the 3 images side by side
    images = [image, generated_image]
    # images = [transforms.ToPILImage()(img) for img in images]
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)
    new_im = Image.new("RGB", (total_width, max_height))
    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    new_im.save(f"output_correct_{i+4}.jpg")
    print("Saved", input_image.replace("ur5e_rgb", "results").replace(".jpg", "_result.jpg"))
