import os
import random
import json
def create_split(base_dir, split_ratio=0):
    image_dir = os.path.join(base_dir)
    files = os.listdir(image_dir)
    image_list = {'rgb': [],
                  'mask': []}

    for f in files:
        if os.path.isdir(os.path.join(base_dir, f)) and "view" not in f:
            for q in os.listdir(os.path.join(image_dir, f)):
                if 'rgb' in str(q):
                    for q_ in os.listdir(os.path.join(image_dir, f, q)):
                        # for q__ in os.listdir(os.path.join(image_dir, f, q, q_)):
                        image_list['rgb'].append(os.path.join(image_dir, f, q, q_))
                        image_list['mask'].append(os.path.join(image_dir, f, q, q_).replace('rgb', 'mask'))
                        print(os.path.join(image_dir, f, q, q_))
                    
    # Randomly shuffle the list of files
    # random.shuffle(image_list)
    
    # Calculate the split index
    # num_files = len(files)
    # split_index = int(num_files * split_ratio)
    
    # # Split the files into training and validation sets
    # validation_files = files[:split_index]
    # train_files = image_list[split_index:]
    
    # Write the filenames to their respective files
    with open(os.path.join(base_dir, 'train_franka_finetune_gripper.json'), 'w') as f:
        json.dump(image_list, f)

    # with open(os.path.join(base_dir, 'val.txt'), 'w') as f:
    #     for file in validation_files:
    #         # Write only the basename without extension
    #         f.write(os.path.splitext(file)[0] + '\n')

# Usage
base_dir = '/rscratch/cfxu/diffusion-RL/style-transfer/data/sim_rollout_images'  # Update this to your actual directory
create_split(base_dir, split_ratio=0)  # Approximately 1:5 ratio
