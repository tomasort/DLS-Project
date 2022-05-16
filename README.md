# DLS-Project
Deep Learning Systems Final Project

# Implementation

In this repository, we store the code for our Deep Learning Systems Project. In this project we try to estimate the velocity of a vehicle from a single camera from inside the car. 

## Requirements

We can run the program in this repo using python 3. 
The requirements are only: 
```
pytorch
torchvision
imageio
opencv-python
```

## How to run

We can run the training script like so: 
``python train_model.py --dataset_mode of --dataset ./regular_frames --model efficientnet --batch 64 --save_path ./results --epoch 100 --lr 0.01``


## Data

We got the data from the [speed challenge](https://github.com/commaai/speedchallenge) which was a competition held by comma.ai in 2017. This data can be found the comma.ai repo. The format is supposed to mmp4 files.
One for testing and the other for training. 

### Optical Flow
We used optical flow to capture the movement between frames. Optical Flow can be defined as the distribution of apparent velocities of movement of brightness pattern in an image. We used OpenCV to calculate the optical flow. 
The algorithm that we employed is the Farneback method. This method computes the Dense optical flow. That means it computes the optical flow from each pixel point in the current image to each pixel point in the next image.

## Model



# Results
this should contain evaluation results. 
