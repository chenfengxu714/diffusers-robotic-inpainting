import os
import numpy as np
import argparse
import random
import torch
import torch.backends.cudnn as cudnn
from importlib import import_module
from segment_anything import sam_model_registry

class InferenceManager:
    def __init__(self, args):
        self.args = args
        self.setup_seed()
        # self.setup_directories()
        self.setup_model()

    def setup_seed(self):
        if not self.args['deterministic']:
            cudnn.benchmark = True
            cudnn.deterministic = False
        else:
            cudnn.benchmark = False
            cudnn.deterministic = True
        seed = self.args['seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def setup_directories(self):
        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir)

        if self.args.is_savenii:
            self.test_save_path = os.path.join(self.args.output_dir, 'masks')
            os.makedirs(self.test_save_path, exist_ok=True)
            self.test_saver2r_path = os.path.join(self.args.output_dir, 'r2r_images')
            os.makedirs(self.test_saver2r_path, exist_ok=True)
        else:
            self.test_save_path = None
            self.test_saver2r_path = None

    def setup_model(self):
        sam, img_embedding_size = sam_model_registry[self.args['vit_name']](
            image_size=self.args['img_size'],
            num_classes=self.args['num_classes'],
            checkpoint=self.args['ckpt']
        )
        sam = sam.to(self.args['device'])
        pkg = import_module(self.args['module'])
        self.model = pkg.LoRA_Sam(sam, self.args['rank'])

        assert self.args['lora_ckpt'] is not None
        self.model.load_lora_parameters(self.args['lora_ckpt'])
        self.model = self.model.to(self.args['device'])

        self.multimask_output = self.args['num_classes'] > 1
        self.model.eval()

    def inference(self, images):
        with torch.no_grad():
            images = torch.nn.functional.upsample_bilinear(images.to(self.args['device']).float() / 1.0, size=(self.args['img_size'], self.args['img_size']))
            outputs = self.model(images, self.multimask_output, self.args['img_size'])

            output_masks = outputs['masks']
            output_masks = torch.nn.functional.upsample_bilinear(output_masks.float(), size=(self.args['img_size'], self.args['img_size']))
            output_masks = (output_masks[:, 1, :, :] > self.args['mask_threshold']).unsqueeze(1)

            masked_image = (images / 255.0) * output_masks

            # save_image(output_masks.float(), os.path.join(self.test_save_path, file[0]))
            output_masks = output_masks.expand_as(images)
            masked_image[output_masks == 0] = 1
            # save_image(masked_image, os.path.join(self.test_saver2r_path, file[0]))
            return masked_image, output_masks

    @staticmethod
    def config_to_dict(config):
        items_dict = {}
        with open(config, 'r') as f:
            items = f.readlines()
        for item in items:
            key, value = item.strip().split(': ')
            items_dict[key] = value
        return items_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='The config file provided by the trained model')
    # parser.add_argument('--image_dir', type=str, default='testset/test_vol_h5/')
    parser.add_argument('--dataset', type=str, default='Robot', help='Experiment name')
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default='/output')
    parser.add_argument('--img_size', type=int, default=256, help='Input image size of the network')
    parser.add_argument('--input_size', type=int, default=256, help='The input size for training SAM model')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--is_savenii', action='store_true', help='Whether to save results during inference')
    parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
    parser.add_argument('--ckpt', type=str, default='./checkpoints/sam_vit_h_4b8939.pth', help='Pretrained checkpoint')
    parser.add_argument('--lora_ckpt', type=str, default='checkpoints/epoch_1.pth', help='The checkpoint from LoRA')
    parser.add_argument('--vit_name', type=str, default='vit_h', help='Select one vit model')
    parser.add_argument('--rank', type=int, default=4, help='Rank for LoRA adaptation')
    parser.add_argument('--module', type=str, default='sam_lora_image_encoder')
    parser.add_argument('--mask_threshold', type=float, default=0.5)
    args = parser.parse_args()

    if args.config is not None:
        config_dict = InferenceManager.config_to_dict(args.config)
        for key in config_dict:
            setattr(args, key, config_dict[key])

    manager = InferenceManager(args)
    manager.inference()

if __name__ == '__main__':
    main()
