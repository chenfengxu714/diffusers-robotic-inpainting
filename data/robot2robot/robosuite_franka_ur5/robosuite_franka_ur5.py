import pandas as pd
# from huggingface_hub import hf_hub_PATH
import datasets
import os

_VERSION = datasets.Version("0.0.2")

_DESCRIPTION = "TODO"
_HOMEPAGE = "TODO"
_LICENSE = "TODO"
_CITATION = "TODO"

# For masked inpainting, the 4 images are
# "image" (ground truth image): franka_rgb
# "input_image" (image to be masked): ur5_rgb
# "mask" (mask): union of ur5_mask and franka_mask will be 0s (masked), everything else will be 1s (not masked)
# "conditioning_image" (image to be masked): ur5_rgb

# For analytic inpainting improvement, the 4 images are
# "image" (ground truth image): franka_rgb
# "input_image" (analytically inpainted): franka_analytic_inpainted
# "mask" (mask): all 0s (everything is black, meaning not masked)
# "conditioning_image" (analytically inpainted): franka_analytic_inpainted

# For img2img translation, the 4 images are
# "image" (ground truth image): franka_rgb
# "input_image" (image to be masked): ur5_rgb
# "mask" (mask): all 0s (everything is black, meaning not masked)
# "conditioning_image" (image to be masked): ur5_rgb

_FEATURES = datasets.Features(
    {
        "image": datasets.Image(),
        "input_image": datasets.Image(),
        # "mask": datasets.Image(),
        # "conditioning_image": datasets.Image(),
        "text": datasets.Value("string"),
    },
)

METADATA_PATH = "/home/lawrence/xembody_followup/diffusers-robotic-inpainting/data/robot2robot/robosuite_franka_ur5/paired_images.jsonl"

# 1. image is always the ground truth image
IMAGES_PATH = "/home/lawrence/xembody_followup/robot2robot/rendering/paired_images_0/ur5e_rgb"

# 2. input_image can be the target robot before masking or the analytically inpainted image
INPUT_IMAGES_PATH = "/home/lawrence/xembody_followup/robot2robot/rendering/paired_images_0/panda_rgb"

# 3. mask is the union of the target robot mask and the ground truth robot mask or all 0s
MASKS_PATH = "/home/lawrence/xembody_followup/robot2robot/rendering/paired_images_0/panda_mask"

# 4. conditioning_image can be the target robot before masking or the analytically inpainted image
CONDITIONING_IMAGES_PATH = INPUT_IMAGES_PATH

_DEFAULT_CONFIG = datasets.BuilderConfig(name="default", version=_VERSION)


class Fill50k(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [_DEFAULT_CONFIG]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=_FEATURES,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        metadata_path = METADATA_PATH
        images_dir = IMAGES_PATH
        input_image_dir = INPUT_IMAGES_PATH
        mask_dir = MASKS_PATH
        conditioning_images_dir = CONDITIONING_IMAGES_PATH

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "metadata_path": metadata_path,
                    "images_dir": images_dir,
                    "input_image_dir": input_image_dir,
                    "mask_dir": mask_dir,
                    "conditioning_images_dir": conditioning_images_dir,
                },
            ),
        ]

    def _generate_examples(self, metadata_path, images_dir, input_image_dir, mask_dir, conditioning_images_dir):
        metadata = pd.read_json(metadata_path, lines=True)

        for _, row in metadata.iterrows():
            text = row["text"]

            image_path = row["image"]
            # image_path = os.path.join(images_dir, image_path)
            image = open(image_path, "rb").read()

            input_image_path = row["input_image"]
            # input_image_path = os.path.join(input_image_dir, input_image_path)
            input_image = open(input_image_path, "rb").read()
            
            # mask_path = row["mask"]
            # # mask_path = os.path.join(mask_dir, mask_path)
            # mask = open(mask_path, "rb").read()


            # conditioning_image_path = row["conditioning_image"]
            # # conditioning_image_path = os.path.join(conditioning_images_dir, conditioning_image_path)
            # conditioning_image = open(conditioning_image_path, "rb").read()

            
            yield row["image"], {
                "text": text,
                "image": {
                    "path": image_path,
                    "bytes": image,
                },
                "input_image": {
                    "path": input_image_path,
                    "bytes": input_image,
                },
                # "mask": {
                #     "path": mask_path,
                #     "bytes": mask,
                # },
                # "conditioning_image": {
                #     "path": conditioning_image_path,
                #     "bytes": conditioning_image,
                # },
            }
