export MODEL_DIR="runwayml/stable-diffusion-inpainting"
export CONTROLNET_MODEL="lllyasviel/control_v11p_sd15_inpaint" 
export DATA_DIR="/home/lawrence/diffusers-robotic-inpainting/data/success_trajs_withposeanddepth_256"
export OUTPUT_DIR="/shared/projects/diffusers/outputs/test"

CUDA_VISIBLE_DEVICES=0 accelerate launch train_controlnet_robotics.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --controlnet_model_name_or_path=$CONTROLNET_MODEL \
 --resume_from_checkpoint="latest" \
 --output_dir=$OUTPUT_DIR \
 --train_data_dir=$DATA_DIR \
 --image_column="image" \
 --input_image_column="input_image" \
 --mask_column="mask" \
 --conditioning_image_column="conditioning_image" \
 --caption_column="text" \
 --use_condition_as_input_image=True \
 --resolution=256 \
 --max_train_steps=10000 \
 --learning_rate=5e-5 \
 --validation_image "/home/lawrence/diffusers/data/success_trajs_withposeanddepth_256/ur5e_rgb/1/34.jpg" \
 --validation_mask "/home/lawrence/diffusers/data/success_trajs_withposeanddepth_256/dummy_mask/0.jpg" \
 --validation_prompt "create a high quality image with a Franka Panda robot, a table, and a red cube on the table" \
 --validation_steps=25 \
 --train_batch_size=64 \
 --gradient_accumulation_steps=8 \
 --report_to="tensorboard" \
 --checkpointing_steps=200
#  --checkpoints_total_limit=5 \
