#!/bin/bash
#SBATCH --job-name=project-efficientnet-two-frames-per-sample
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --output=%x.out
#SBATCH --mem=32GB
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:v100:1

module purge
source /scratch/tor213/.env/bin/activate
cd /scratch/tor213/DLS-Project/

python train_model.py --dataset_mode d --dataset ./regular_frames --model efficientnet --batch 32