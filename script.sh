#!/bin/bash

# Request half an hour of runtime:
#SBATCH --time=20:00:00

# Ask for the GPU partition and 1 GPU
#SBATCH -p gpu --gres=gpu:1 --gres-flags=enforce-binding

# Default resources are 1 core with 2.8GB of memory.

# Use more memory (4GB) (CPU RAM):
#SBATCH --mem=20G

# Specify a job name:
#SBATCH -J audioDiffusionTraining

# Specify an output file
#SBATCH -o audioDiffusion.out
#SBATCH -e audioDiffusion.out

# Email reports
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=nuo_wen_lei@brown.edu


# Set up the environment by loading modules
module load python/3.9.0
module load cuda/8.0.61 cudnn/5.1
source audioTrain/bin/activate

# Run a script
python3 ccv_trainer.py