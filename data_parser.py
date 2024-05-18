import h5py
import os
import cv2
import numpy as np

def parse_data(file_path, output_path):
    for file in os.listdir(file_path):
        with h5py.File(os.path.join(file_path, file), 'r') as f:
            varied_camera_1_left_image = f['observation']['camera']['image']['varied_camera_1_left_image_transferred']
            for i in range(varied_camera_1_left_image.shape[0]):
                data = varied_camera_1_left_image[i]
                if os.path.exists(os.path.join(output_path, file_path.split('/')[-2], file_path.split('/')[-1])) == False:
                    os.makedirs(os.path.join(output_path, file_path.split('/')[-2], file_path.split('/')[-1]))
                print(os.path.join(output_path, file_path.split('/')[-2], file_path.split('/')[-1], str(i) + '.png'))
                cv2.imwrite(os.path.join(output_path, file_path.split('/')[-2], file_path.split('/')[-1], str(i) + '.png'),  cv2.cvtColor(data.astype(np.uint8), cv2.COLOR_BGR2RGB))
    # return data

root_path = "/rscratch/cfxu/diffusion-RL/style-transfer/data/output_h5"
output_path = "/rscratch/cfxu/diffusion-RL/style-transfer/data/parsered_images_robo_transferred"
for file in os.listdir(root_path):
    data = parse_data(os.path.join(root_path, file), output_path)
    # print(data.shape)