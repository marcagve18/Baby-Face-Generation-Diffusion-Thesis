#!/bin/bash
#SBATCH --job-name=diffusion_inference  # Job name
#SBATCH --output=./../../model_logs/ignore.out 
#SBATCH --error=./../../model_logs/inference.log  
#SBATCH --nodes=1  # Number of nodes
#SBATCH --ntasks-per-node=1  # Number of tasks per node
#SBATCH --cpus-per-task=1  # Number of CPU cores per task
#SBATCH --gres=gpu:1  # Number of GPUs required
#SBATCH --mem=20GB  # Memory per node
#SBATCH --time=01:00:00  # Time limit (hours:minutes:seconds)
#SBATCH --partition=medium
 
module load CUDA/11.4.3
module load Anaconda3

source ./../../.env

source activate $HOME_PATH/project/anaconda3/envs/envname

python3 multiple_inference.py --models "sd_15_v3" "sd_15_v4" "sd_15_v5" --prompt "$1" --num_inference_steps 20
