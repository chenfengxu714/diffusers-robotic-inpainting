import os
import sys
from tqdm import tqdm
import logging
import numpy as np
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from utils import test_single_volume
from importlib import import_module
from segment_anything import sam_model_registry
from datasets.robot_test_dataset import Test_Robot_dataset
from torchvision.utils import save_image


class_to_name = {1: 'spleen', 2: 'right kidney', 3: 'left kidney', 4: 'gallbladder', 5: 'liver', 6: 'stomach', 7: 'aorta', 8: 'pancreas'}


def inference(args, multimask_output, db_config, model, test_save_path=None, test_saver2r_path=None):
    db_test = db_config['Dataset'](image_dir=args.image_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info(f'{len(testloader)} test iterations per epoch')
    model.eval()
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        image = sampled_batch['image'].cuda()
        file = sampled_batch['file_name']
        image = torch.nn.functional.upsample_bilinear(image.float()/1.0, size=(args.img_size, args.img_size))
        outputs = model(image, multimask_output, args.img_size)
        
        output_masks = outputs['masks']
        
        output_masks =  torch.nn.functional.upsample_bilinear(output_masks.float(), size=(args.img_size, args.img_size))
        # output_masks = output_masks > 0.9
        # print(output_masks.shape)
        # output_masks = torch.argmax(torch.softmax(output_masks, dim=1), dim=1, keepdim=True)
        output_masks = (output_masks[:, 1, :, :] > args.mask_threshold).unsqueeze(1)
        # masked_image[ == 1] = 1.0
        # image = torch.nn.functional.upsample_bilinear(image.float()/255.0, size=(224, 224))
        masked_image = (image/255.0) * output_masks
        # Saving the original images and masks
        
        # concatenated_image = torch.cat((image/255.0, masked_image), dim=3)
        save_image(output_masks.float(), os.path.join(test_save_path, file[0]))
        # Creating and saving the masked images
         # Apply mask to the image
        # print(concatenated_image.shape)
        # save_image(concatenated_image, os.path.join(test_save_path, f'concatenated_{i_batch}.png'))
        output_masks = output_masks.expand_as(image)
        masked_image[output_masks==0] = 1
        save_image(masked_image, os.path.join(test_saver2r_path, file[0]))
        
        # Optionally: Log progress
        logging.info(f"Processed and saved batch {i_batch + 1}/{len(testloader)}")

def config_to_dict(config):
    items_dict = {}
    with open(config, 'r') as f:
        items = f.readlines()
    for i in range(len(items)):
        key, value = items[i].strip().split(': ')
        items_dict[key] = value
    return items_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='The config file provided by the trained model')
    parser.add_argument('--image_dir', type=str, default='testset/test_vol_h5/')
    parser.add_argument('--dataset', type=str, default='Robot', help='Experiment name')
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default='/output')
    parser.add_argument('--img_size', type=int, default=256, help='Input image size of the network')
    parser.add_argument('--input_size', type=int, default=256, help='The input size for training SAM model')
    parser.add_argument('--seed', type=int,
                        default=1234, help='random seed')
    parser.add_argument('--is_savenii', action='store_true', help='Whether to save results during inference')
    parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
    parser.add_argument('--ckpt', type=str, default='./checkpoints/sam_vit_h_4b8939.pth',
                        help='Pretrained checkpoint')
    parser.add_argument('--lora_ckpt', type=str, default='checkpoints/epoch_1.pth', help='The checkpoint from LoRA')
    parser.add_argument('--vit_name', type=str, default='vit_h', help='Select one vit model')
    parser.add_argument('--rank', type=int, default=4, help='Rank for LoRA adaptation')
    parser.add_argument('--module', type=str, default='sam_lora_image_encoder')
    parser.add_argument('--mask_threshold', type=float, default=0.5)
    args = parser.parse_args()

    if args.config is not None:
        # overwtite default configurations with config file\
        config_dict = config_to_dict(args.config)
        for key in config_dict:
            setattr(args, key, config_dict[key])

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    dataset_config = {
        'Robot': {
            'Dataset': Test_Robot_dataset,
            'num_classes': args.num_classes,
        }
    }
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # register model
    sam, img_embedding_size = sam_model_registry[args.vit_name](image_size=args.img_size,
                                                                    num_classes=args.num_classes,
                                                                    checkpoint=args.ckpt)
    
    pkg = import_module(args.module)
    net = pkg.LoRA_Sam(sam, args.rank).cuda()

    assert args.lora_ckpt is not None
    net.load_lora_parameters(args.lora_ckpt)

    if args.num_classes > 1:
        multimask_output = True
    else:
        multimask_output = False

    # initialize log
    # log_folder = os.path.join(args.output_dir, 'test_log')
    # os.makedirs(log_folder, exist_ok=True)
    # logging.basicConfig(filename=log_folder + '/' + 'log.txt', level=logging.INFO,
    #                     format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    # logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    # logging.info(str(args))

    if args.is_savenii:
        test_save_path = os.path.join(args.output_dir, 'masks')
        os.makedirs(test_save_path, exist_ok=True)
        test_saver2r_path = os.path.join(args.output_dir, 'r2r_images')
        os.makedirs(test_saver2r_path, exist_ok=True)
    else:
        test_save_path = None
    inference(args, multimask_output, dataset_config[dataset_name], net, test_save_path, test_saver2r_path)
