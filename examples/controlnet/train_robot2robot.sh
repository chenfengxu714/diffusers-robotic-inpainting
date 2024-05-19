# export MODEL_DIR="runwayml/stable-diffusion-v1-5"
# # export CONTROLNET_MODEL="/shared/projects/diffusers/outputs/franka_to_ur5_diverse_angles/lr_1e-4_bs_512/checkpoint-12450/controlnet" # a good model for franka to ur5 base
# export CONTROLNET_MODEL="/shared/projects/diffusers/outputs/franka_to_ur5_diverse_angles_viola_finetune/lr_5e-5_bs_512/checkpoint-18550/controlnet" # last checkpoint after finetuning on the viola dataset
# export DATA_DIR="/home/lawrence/xembody_followup/diffusers-robotic-inpainting/data/robot2robot/mirage_finetune"
# export OUTPUT_DIR="/shared/projects/diffusers/outputs/franka_to_ur5_diverse_angles_mirage_finetune/lr_5e-5_bs_512"
# # validation_image "/home/lawrence/xembody_followup/robot2robot/rendering/paired_images_8/panda_rgb/2339/1.jpg" \
# # validation_image /home/lawrence/xembody_followup/viola_dataset/plateforkdomain/168_sim.png
# CUDA_VISIBLE_DEVICES=4 accelerate launch train_controlnet.py \
#  --pretrained_model_name_or_path=$MODEL_DIR \
#  --controlnet_model_name_or_path=$CONTROLNET_MODEL \
#  --resume_from_checkpoint="latest" \
#  --output_dir=$OUTPUT_DIR \
#  --train_data_dir=$DATA_DIR \
#  --image_column="image" \
#  --conditioning_image_column="input_image" \
#  --caption_column="text" \
#  --resolution=256 \
#  --max_train_steps=20000 \
#  --learning_rate=5e-5 \
#  --validation_image "/home/lawrence/xembody_followup/mirage_data/test1_cropped_resized.png" \
#  --validation_prompt "create a high quality image with a UR5 robot and white background" \
#  --validation_steps=25 \
#  --train_batch_size=32 \
#  --gradient_accumulation_steps=16 \
#  --report_to="tensorboard" \
#  --checkpointing_steps=50
# #  --checkpoints_total_limit=5 \


# franka to ur5
export MODEL_DIR="runwayml/stable-diffusion-v1-5"
export DATA_DIR="/home/lawrence/xembody_followup/diffusers-robotic-inpainting/data/robot2robot/franka_ur5_all_"
export OUTPUT_DIR="/shared/projects/diffusers/outputs/franka_to_ur5_diverse_angles_all/lr_1e-4_bs_512"
CUDA_VISIBLE_DEVICES=0 accelerate launch train_controlnet.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --resume_from_checkpoint="latest" \
 --output_dir=$OUTPUT_DIR \
 --train_data_dir=$DATA_DIR \
 --image_column="image" \
 --conditioning_image_column="input_image" \
 --caption_column="text" \
 --resolution=256 \
 --max_train_steps=20000 \
 --learning_rate=1e-4 \
 --validation_image "/home/lawrence/xembody_followup/mirage_data/test4_cropped_resized.png" \
 --validation_prompt "create a high quality image with a UR5 robot and white background" \
 --validation_steps=25 \
 --train_batch_size=32 \
 --gradient_accumulation_steps=16 \
 --report_to="tensorboard" \
 --checkpointing_steps=50


# ur5 to franka
# export MODEL_DIR="runwayml/stable-diffusion-v1-5"
# export DATA_DIR="/home/lawrence/xembody_followup/diffusers-robotic-inpainting/data/robot2robot/ur5_franka_all_"
# export OUTPUT_DIR="/shared/projects/diffusers/outputs/ur5_to_franka_diverse_angles_all/lr_1e-4_bs_512"
# CUDA_VISIBLE_DEVICES=1 accelerate launch train_controlnet.py \
#  --pretrained_model_name_or_path=$MODEL_DIR \
#  --resume_from_checkpoint="latest" \
#  --output_dir=$OUTPUT_DIR \
#  --train_data_dir=$DATA_DIR \
#  --image_column="image" \
#  --conditioning_image_column="input_image" \
#  --caption_column="text" \
#  --resolution=256 \
#  --max_train_steps=20000 \
#  --learning_rate=1e-4 \
#  --validation_image "/home/lawrence/xembody_followup/mirage_data/test5_cropped_resized.png" \
#  --validation_prompt "create a high quality image with a Franka robot and white background" \
#  --validation_steps=25 \
#  --train_batch_size=32 \
#  --gradient_accumulation_steps=16 \
#  --report_to="tensorboard" \
#  --checkpointing_steps=50