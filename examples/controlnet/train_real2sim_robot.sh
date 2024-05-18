export MODEL_DIR="runwayml/stable-diffusion-v1-5"
export DATA_DIR="/rscratch/cfxu/diffusion-RL/style-transfer/diffusers-robotic-inpainting/data/real2sim/robosuite_real2sim.py"
export OUTPUT_DIR="/rscratch/cfxu/diffusion-RL/style-transfer/diffusers-robotic-inpainting/outputs/real2sim_masks"

CUDA_VISIBLE_DEVICES=3,4 accelerate launch train_controlnet.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --resume_from_checkpoint="latest" \
 --output_dir=$OUTPUT_DIR \
 --train_data_dir=$DATA_DIR \
 --image_column="image" \
 --conditioning_image_column="input_image" \
 --caption_column="text" \
 --resolution=256 \
 --max_train_steps=10000 \
 --learning_rate=1e-4 \
 --validation_image "/rscratch/cfxu/diffusion-RL/style-transfer/diffusers-robotic-inpainting/data/bowldomain_masked_images/168.png" "/rscratch/cfxu/diffusion-RL/style-transfer/diffusers-robotic-inpainting/data/bowldomain_masked_images/350.png" \
 --validation_prompt "transfering the real style to simulation style" "transfering the real style to simulation style" \
 --validation_steps=25 \
 --train_batch_size=16\
 --gradient_accumulation_steps=16 \
 --report_to="tensorboard" \
 --enable_xformers_memory_efficient_attention \
 --checkpointing_steps=500 \
 
#  --checkpoints_total_limit=5 \
