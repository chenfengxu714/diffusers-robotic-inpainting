import os
import json

def create_split(base_dir, split_ratio=0):
    image_dir = os.path.join(base_dir)
    files = os.listdir(image_dir)
    image_list = {'jaco': {'rgb': [], 'mask': []},
                  'panda': {'rgb': [], 'mask': []},
                  'sawyer': {'rgb': [], 'mask': []},
                  'ur5e': {'rgb': [], 'mask': []}}

    for f in files:
        if os.path.isdir(os.path.join(base_dir, f)):
            for q in os.listdir(os.path.join(image_dir, f)):
                for keyword in image_list.keys():
                    if keyword in str(q) and 'rgb' in str(q):
                        for q_ in os.listdir(os.path.join(image_dir, f, q)):
                            if len(os.listdir(os.path.join(image_dir, f, q, q_))) == 0:
                                continue
                            else:
                                for q__ in os.listdir(os.path.join(image_dir, f, q, q_)):
                                    if 'rgb' in os.path.join(image_dir, f, q, q_) and not 'brightness' in os.path.join(image_dir, f, q, q_):
                                        image_list[keyword]['rgb'].append(os.path.join(image_dir, f, q, q_, q__))
                                        image_list[keyword]['mask'].append(os.path.join(image_dir, f, q, q_, q__).replace('rgb', 'mask'))
                                    # # Replace rgb with mask, also replace rgb_brightness_augmented with mask
                                    # if 'rgb_brightness_augmented' in os.path.join(image_dir, f, q, q_, q__):
                                    #     image_list[keyword]['mask'].append(os.path.join(image_dir, f, q, q_, q__).replace('rgb_brightness_augmented', 'mask'))
                                    # elif 'rgb' in os.path.join(image_dir, f, q, q_):
                                    #     image_list[keyword]['mask'].append(os.path.join(image_dir, f, q, q_, q__).replace('rgb', 'mask'))
                                        print(os.path.join(image_dir, f, q, q_, q__))

    # Write the filenames to their respective files
    for keyword in image_list.keys():
        with open(os.path.join(base_dir, f'train_{keyword}_finetune_gripper.json'), 'w') as f:
            json.dump(image_list[keyword], f)

# Usage
base_dir = '/rscratch/cfxu/diffusion-RL/style-transfer/data/franka_ur5_sawyer_jaco'  # Update this to your actual directory
create_split(base_dir, split_ratio=0)
