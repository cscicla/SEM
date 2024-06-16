#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --time=0-10:00:00
#SBATCH --error=sem1results-%J.err
#SBATCH --output=sem1results-%J.out
#SBATCH --job-name=sem1
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu32

module load anaconda/anaconda3
module load cuda/cuda-10.2

export python="/home/crcvreu.student10/my-envs/semenv/bin/python"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/crcvreu.student10/my-envs/semenv/$

source activate semenv