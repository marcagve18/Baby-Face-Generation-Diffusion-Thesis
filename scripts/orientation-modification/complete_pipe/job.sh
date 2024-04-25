#!/bin/bash
#SBATCH --job-name=diffusion_inference  # Job name
#SBATCH --output=./../../../model_logs/ignore.out 
#SBATCH --error=./../../../model_logs/inference.log  
#SBATCH --nodes=1  # Number of nodes
#SBATCH --ntasks-per-node=5  # Number of tasks per node
#SBATCH --cpus-per-task=5  # Number of CPU cores per task
#SBATCH --gres=gpu:1  # Number of GPUs required
#SBATCH --mem=50GB  # Memory per node
#SBATCH --time=01:00:00  # Time limit (hours:minutes:seconds)
#SBATCH --partition=medium
 
module load CUDA/11.4.3
module load Anaconda3

source ./../../../.env

source activate $HOME_PATH/project/anaconda3/envs/envname

CUDA_LAUNCH_BLOCKING=1 python3 inference.py --path="/home/maguilar/TFG/Baby-Face-Generation-Diffusion-Thesis/output/images/thesis/generation/individuals/img_2024-03-07 12:55:52.994781_Frontal portrait of a smiling asian baby.png"
