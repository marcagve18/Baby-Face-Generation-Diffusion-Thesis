#!/bin/bash
#SBATCH --job-name=caption_inference  # Job name
#SBATCH --output=./../../../model_logs/ignore.out 
#SBATCH --error=./../../../model_logs/inference.log  
#SBATCH --nodes=1  # Number of nodes
#SBATCH --ntasks-per-node=1  # Number of tasks per node
#SBATCH --cpus-per-task=1  # Number of CPU cores per task
#SBATCH --gres=gpu:quadro:1  # Number of GPUs required
#SBATCH --mem=100GB  # Memory per node
#SBATCH --time=24:00:00  # Time limit (hours:minutes:seconds)
#SBATCH --partition=high
 
module load CUDA/11.4.3
module load Anaconda3

source ./../../../.env

source activate $HOME_PATH/project/anaconda3/envs/envname

CUDA_LAUNCH_BLOCKING=1 python3 caption_dataset.py
