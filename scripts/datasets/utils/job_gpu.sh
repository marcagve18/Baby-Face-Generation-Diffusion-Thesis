#!/bin/bash
#SBATCH --job-name=diffusion_training
#SBATCH --output=./../../../model_logs/ignore.out 
#SBATCH --error=./../../../model_logs/dataset_check.log  
#SBATCH --nodes=1  # Number of nodes
#SBATCH --ntasks-per-node=1  # Number of tasks per node
#SBATCH --cpus-per-task=1  # Number of CPU cores per task
#SBATCH --mem=100GB  # Memory per node
#SBATCH --time=24:00:00  # Time limit (hours:minutes:seconds)
#SBATCH --partition=high
 

module load CUDA/11.4.3
module load Anaconda3

source ./../../../.env

source activate $HOME_PATH/project/anaconda3/envs/envname

python3 count.py