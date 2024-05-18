import os
import subprocess

def run_command_for_each_folder(base_dir, base_output_dir, lora_ckpt):
    # Iterate over each folder in the base directory
    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        if os.path.isdir(folder_path):
            # Construct the output directory path
            output_dir = os.path.join(base_output_dir, folder_name)
            # Create the output directory if it doesn't exist
            if os.path.isdir(output_dir):
                print(f"Output directory {output_dir} already exists. Skipping...")
                continue
            os.makedirs(output_dir, exist_ok=True)
            
            # Construct the command
            command = (
                f"CUDA_VISIBLE_DEVICES=3 python test_robot.py --image_dir {folder_path} "
                f"--is_savenii --output_dir {output_dir} --lora_ckpt {lora_ckpt}"
            )
            print(f"Running command: {command}")
            
            # Run the command
            subprocess.run(command, shell=True, check=True)

if __name__ == "__main__":
    base_dir = "/rscratch/cfxu/diffusion-RL/style-transfer/data/parsered_images_robo/2024-05-10-tiger-franka-gripper"
    base_output_dir = "/rscratch/cfxu/diffusion-RL/style-transfer/data/parsered_images_robo/2024-05-10-tiger-franka-gripper_masks"
    lora_ckpt = "finetuned_lora_wo_pretrain_gripper/Robot_256_pretrain_vit_h_epo400_bs64_lr0.0002/epoch_0.pth"
    
    run_command_for_each_folder(base_dir, base_output_dir, lora_ckpt)
