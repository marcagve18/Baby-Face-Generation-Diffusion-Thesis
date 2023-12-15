#!/bin/bash
#SBATCH --job-name=diffusion_inference  # Job name
#SBATCH --output=./../../model_logs/ignore.out 
#SBATCH --error=./../../model_logs/inference.log  
#SBATCH --nodes=4  # Number of nodes
#SBATCH --ntasks-per-node=4  # Number of tasks per node
#SBATCH --cpus-per-task=4  # Number of CPU cores per task
#SBATCH --gres=gpu:1  # Number of GPUs required
#SBATCH --mem=100GB  # Memory per node
#SBATCH --time=00:05:00  # Time limit (hours:minutes:seconds)
#SBATCH --partition=high
 
module load CUDA/11.4.3
module load Anaconda3

source ./../../.env

source activate $HOME_PATH/project/anaconda3/envs/envname

python3 multiple_inference.py --models "flickr-first-model" "flickr-second-model" "flickr-third-model" "flickr-fifth-model" "flickr-sixth-model" --prompt "$1" --num_inference_steps 50
