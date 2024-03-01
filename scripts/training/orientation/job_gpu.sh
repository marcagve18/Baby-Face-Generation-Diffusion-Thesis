#!/bin/bash
#SBATCH --job-name=diffusion_training
#SBATCH --output=./../../../model_logs/ignore.out 
#SBATCH --error=./../../../model_logs/controlnet_training_%j.log  
#SBATCH --nodes=1  # Number of nodes
#SBATCH --ntasks-per-node=1  # Number of tasks per node
#SBATCH --cpus-per-task=1  # Number of CPU cores per task
#SBATCH --gres=gpu:2  # Number of GPUs required
#SBATCH --mem=10GB  # Memory per node
#SBATCH --time=24:00:00  # Time limit (hours:minutes:seconds)
#SBATCH --partition=high
#SBATCH --mail-type=ALL
#SBATCH --mail-user=marcaguilar.1803@gmail.com
 
# Check if $1 is not empty
if [ -z "$1" ]; then
  echo "Error: Model name not provided. Usage: $0 <model_name>"
  exit 1
fi

module load CUDA/11.4.3
module load Anaconda3

nvidia-smi

source ./../../../.env

source activate $HOME_PATH/project/anaconda3/envs/envname
accelerate config default

accelerate launch --mixed_precision="fp16" --multi_gpu --num_processes=2 train_controlnet.py  \
  --pretrained_model_name_or_path="/home/maguilar/TFG/Baby-Face-Generation-Diffusion-Thesis/models/sd_15_v5" \
  --dataset_name="/home/maguilar/TFG/Baby-Face-Generation-Diffusion-Thesis/datasets/orientation/v1" \
  --conditioning_image_column=conditioning_image \
  --image_column=image \
  --caption_column=prompt \
  --resolution=512  \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --max_train_steps=30000 \
  --learning_rate=1e-05 \
  --checkpointing_steps=2000 \
  --max_grad_norm=1 \
  --use_8bit_adam \
  --enable_xformers_memory_efficient_attention \
  --lr_scheduler="cosine" --lr_warmup_steps=0 \
  --output_dir="/home/maguilar/TFG/Baby-Face-Generation-Diffusion-Thesis/models/controlnets/$1" \
