#!/bin/bash
#SBATCH --job-name=diffusion_training
#SBATCH --output=./../../model_logs/ignore.out 
#SBATCH --error=./../../model_logs/diffusion_training_%j.log  
#SBATCH --nodes=1  # Number of nodes
#SBATCH --ntasks-per-node=1  # Number of tasks per node
#SBATCH --cpus-per-task=1  # Number of CPU cores per task
#SBATCH --gres=gpu:quadro:2  # Number of GPUs required
#SBATCH --mem=100GB  # Memory per node
#SBATCH --time=48:00:00  # Time limit (hours:minutes:seconds)
#SBATCH --partition=high
 
# Check if $1 is not empty
if [ -z "$1" ]; then
  echo "Error: Model name not provided. Usage: $0 <model_name>"
  exit 1
fi

module load CUDA/11.4.3
module load Anaconda3

nvidia-smi

source ./../../.env

source activate $HOME_PATH/project/anaconda3/envs/envname
accelerate config default

accelerate launch --mixed_precision="fp16" --multi_gpu --num_processes=2 train_multicaption.py  \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --dataset_name="/home/maguilar/TFG/Baby-Face-Generation-Diffusion-Thesis/datasets/processed/alexg99-captioned_flickr_faces-custom_caption_v2" \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --max_train_steps=120000 \
  --learning_rate=1e-04 \
  --checkpointing_steps=2000 \
  --max_grad_norm=1 \
  --use_8bit_adam \
  --snr_gamma=5.0 \
  --enable_xformers_memory_efficient_attention \
  --lr_scheduler="cosine" --lr_warmup_steps=0 \
  --output_dir="../../models/$1" \
