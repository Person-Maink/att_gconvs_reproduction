#!/bin/bash

#SBATCH --job-name=setup-task-ist
#SBATCH --partition=gpu-a100
#SBATCH --time=00:45:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --account=education-eemcs-courses-dsait4095

# Load modules:
module load 2024r1
module load miniconda3
module load cuda/11.6
module load openmpi
# module load py-torch/1.12.1
# module load py-pip
# module load py-numpy
# module load py-pyyaml
# module load py-tqdm
# module load ffmpeg



source ~/.bashrc

eval "$(conda shell.bash hook)"

conda activate torch

echo "Using Python from:"

which python

python -V

/home/mthakur/.conda/envs/torch/bin/python /home/mthakur/att_gconvs_reproduction/experiments/main.py --visualize False
