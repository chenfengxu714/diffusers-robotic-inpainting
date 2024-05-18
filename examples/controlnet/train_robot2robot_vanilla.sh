export MODEL_DIR="runwayml/stable-diffusion-v1-5"
export DATA_DIR="/home/lawrence/xembody_followup/diffusers-robotic-inpainting/data/robot2robot/robosuite_franka_ur5"
export OUTPUT_DIR="/shared/projects/diffusers/outputs/robot2robot/lr_1e-4"

CUDA_VISIBLE_DEVICES=1 accelerate launch train_controlnet.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --resume_from_checkpoint="latest" \
 --output_dir=$OUTPUT_DIR \
 --train_data_dir=$DATA_DIR \
 --image_column="image" \
 --conditioning_image_column="input_image" \
 --caption_column="text" \
 --resolution=256 \
 --max_train_steps=5000 \
 --learning_rate=1e-4 \
 --validation_image "/home/lawrence/xembody_followup/robot2robot/rendering/paired_images_8/panda_rgb/2339/1.jpg" \
 --validation_prompt "create a high quality image with a UR5 robot and white background" \
 --validation_steps=25 \
 --train_batch_size=32 \
 --gradient_accumulation_steps=16 \
 --report_to="tensorboard" \
 --checkpointing_steps=50
#  --checkpoints_total_limit=5 \
