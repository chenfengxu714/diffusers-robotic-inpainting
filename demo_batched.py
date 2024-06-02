import os
import subprocess

def get_video_mask_pairs(base_dir, mask_dir):
    """
    Get pairs of video and mask directories from the base directory.
    Assumes each video has a corresponding mask directory.
    """
    pairs = []
    for root, dirs, files in os.walk(base_dir):
        for dir_name in dirs:
            video_path = os.path.join(root, dir_name)
            mask_path = os.path.join(mask_dir, dir_name, 'masks')
            save_path = os.path.join(mask_dir, dir_name, 'inpainted_frames')
            if os.path.isdir(video_path) and os.path.isdir(mask_path):
                if not os.path.exists(save_path):
                    pairs.append((video_path, mask_path, save_path))
    return pairs

def run_command(video_path, mask_path, ckpt_path, save_frame_dir, width, height):
    """
    Run the demo.py script for a given video and mask pair.
    """
    command = [
        'python', 'demo.py', 
        '--model', 'e2fgvi_hq', 
        '--video', video_path, 
        '--mask', mask_path, 
        '--ckpt', ckpt_path, 
        '--set_size', 
        '--width', str(width), 
        '--height', str(height), 
        '--save_frame', save_frame_dir
    ]
    os.environ['CUDA_VISIBLE_DEVICES'] = str(7)
    print(f"Running command: {' '.join(command)}")
    subprocess.run(command, check=True)

def main():
    base_dir = "/rscratch/cfxu/diffusion-RL/style-transfer/data/cloth_sweeping_images_left2/cloth_sweeping/raw"
    mask_dir = "/rscratch/cfxu/diffusion-RL/style-transfer/data/cloth_sweeping_images_left2_masks/cloth_sweeping/raw"
    ckpt_path = "/rscratch/cfxu/diffusion-RL/style-transfer/E2FGVI/release_model/E2FGVI-HQ-CVPR22.pth"
    save_frame_base_dir = mask_dir
    width, height = 256, 256

    # Get pairs of video and mask directories
    video_mask_pairs = get_video_mask_pairs(base_dir, mask_dir)

    # Process each pair one by one
    for video_path, mask_path, inpaint_path in video_mask_pairs:
        run_command(video_path, mask_path, ckpt_path, inpaint_path, width, height)

if __name__ == "__main__":
    main()
