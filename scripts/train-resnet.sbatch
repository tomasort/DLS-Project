#!/bin/bash
#SBATCH --job-name=project-resnet-optical-flow
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --output=%x.out
#SBATCH --mem=32GB
#SBATCH --time=6:00:00
#SBATCH --gres=gpu

module purge
source /scratch/tor213/.env/bin/activate
cd /scratch/tor213/DLS-Project/

python train_model.py --dataset_mode of --dataset ./dataset --model resnet --batch 32 --save_path ./resnet --epoch 100 --lr 0.01