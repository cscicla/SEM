#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --time=0-10:00:00
#SBATCH --error=myjobresults-%J.err
#SBATCH --output=myjobresults-%J.out
#SBATCH --job-name=unet-multi
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu32

module load cuda/cuda-10.2
module load python/python-3.8.0-gcc-9.1.0

source activate your_env 

python -m train.py