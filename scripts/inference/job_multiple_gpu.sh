#!/bin/bash
#SBATCH --job-name=diffusion_inference  # Job name
#SBATCH --output=./../../model_logs/ignore.out 
#SBATCH --error=./../../model_logs/inference.log  
#SBATCH --nodes=1  # Number of nodes
#SBATCH --ntasks-per-node=5  # Number of tasks per node
#SBATCH --cpus-per-task=5  # Number of CPU cores per task
#SBATCH --gres=gpu:quadro:1  # Number of GPUs required
#SBATCH --mem=100GB  # Memory per node
#SBATCH --time=01:00:00  # Time limit (hours:minutes:seconds)
#SBATCH --partition=high
 
module load CUDA/11.4.3
module load Anaconda3

source ./../../.env

source activate $HOME_PATH/project/anaconda3/envs/envname

python3 multiple_inference.py --models "flickr-first-model" "flickr-second-model" "flickr-fifth-model" "flickr-sixth-model" "custom_dataset_v1" "sd_15_v2" "sd_15_v3" --prompt "$1" --num_inference_steps 100
