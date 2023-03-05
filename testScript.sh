#!/bin/bash

# Request half an hour of runtime:
#SBATCH --time=00:01:00

# Ask for the GPU partition and 1 GPU
#SBATCH -p gpu --gres=gpu:1 --gres-flags=enforce-binding

# Default resources are 1 core with 2.8GB of memory.

# Specify a job name:
#SBATCH -J audioDiffusionTraining

# Specify an output file
#SBATCH -o audioDiffusion1.out
#SBATCH -e audioDiffusion1.out

# Set up the environment by loading modules
module load python/3.9.0
module load cuda/11.1.1 cudnn/8.1.0
source audioTrainer/bin/activate

# Run a script
python3 cudaTestScript.py