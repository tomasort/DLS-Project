import matplotlib.pyplot as plt
import torchvision
import numpy as np
import argparse
import torch
import time
import os
import cv2

def convertToOptical(prev_image, curr_image):
    prev_image_gray = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)
    curr_image_gray = cv2.cvtColor(curr_image, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prev_image_gray, curr_image_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    hsv = np.zeros_like(prev_image)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,1] = 255
    hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    flow_image_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return flow_image_bgr

def process_video(video_path, dest_dataset):
    stream = "video"
    video = torchvision.io.VideoReader(video_path, stream)
    video.get_metadata()

    video.set_current_stream(stream)

    frames = []  # we are going to save the frames here.
    ptss = []  # pts is a presentation timestamp in seconds (float) of each frame
    prev = None

    for i, frame in enumerate(video):
        if i > 1000:
            break
        if prev is None:
            prev = frame['data']
            continue
        optical_flow = convertToOptical(prev.permute(1, 2, 0).cpu().detach().numpy(), frame['data'].permute(1, 2, 0).cpu().detach().numpy())
        cv2.imwrite(os.path.join(dest_dataset, f"{i-1:05}.jpg"), optical_flow)
        prev = frame['data']
        ptss.append(frame['pts'])

    print(f"Total number of frames: {len(ptss)}. Saved in {dest_dataset}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script for converting a video into frames')
    parser.add_argument('--data_path', type=str, default='./data',
                        help='the path of the train and test videos to convert')
    parser.add_argument('--target_dir', type=str, default='./dataset/',
                        help='the path where the frames will be saved')

    args = parser.parse_args()

    train_video_path = os.path.join(args.data_path, "train.mp4")
    test_video_path = os.path.join(args.data_path, "test.mp4")

    os.makedirs(args.target_dir, exist_ok=True)

    train_target_directory = os.path.join(args.target_dir, "train")
    os.makedirs(train_target_directory, exist_ok=True)
    process_video(train_video_path, train_target_directory)

    test_target_directory = os.path.join(args.target_dir, "test")
    os.makedirs(test_target_directory, exist_ok=True)
    process_video(test_video_path, test_target_directory)
