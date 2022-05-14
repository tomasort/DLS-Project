import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
import torch.optim as optim
from efficientnet_pytorch import EfficientNet
from trainer import train_model
from pathlib import Path
import numpy as np
import argparse
import os
from PIL import Image
import torchvision.models as models
import torchvision.transforms.functional as TF

parser = argparse.ArgumentParser(description='Script for converting a video into frames')
parser.add_argument('--dataset', type=str, default='./dataset',
                    help='the path of the train and test videos to convert')
parser.add_argument('--labels', type=str, default='./data',
                    help='directory where labels are located')
parser.add_argument('--lr', type=float, default=0.1,
                    help='learning rate. The default is 0.1')
parser.add_argument('--epochs', default=350, type=int,
                    help='Number of epochs')
parser.add_argument('--optimizer', default='adam', type=str,
                    help='Adam or SGD')
parser.add_argument('--model', default='efficientnet', type=str,
                    help="what model to train")
parser.add_argument('--model_num', default=0, type=int,
                    help='number for efficientnet')
parser.add_argument('--dataset_mode', default="of", type=str,
                    help='dataset mode that we want to use. It could be of for optical flow or d for double frames')
parser.add_argument('--n', default=2, type=int,
                    help='number of frames to use if the dataset is composed of regular frames for each traning sample')

args = parser.parse_args()

# directory with the optical flow images
of_dir = os.path.join(args.dataset)
# labels as txt file
labels_f = os.path.join(args.labels)


class NFrameDataset(Dataset):
    """Dataset for appending the previous N frames as channels to a sample"""
    def __init__(self, frame_dir, label_f, n, transform=None):
        if n < 1: 
            sys.exit("The value of n is not valid")
        self.transform = transform
        self.n = n
        self.len = len(list(Path(frame_dir).glob('*.jpg'))) - (self.n - 1)
        self.frame_dir = frame_dir
        self.label_file = open(label_f).readlines()

    def __len__(self): 
        return self.len

    def __getitem__(self, idx):
        frames = []
        for i in range(idx, idx + self.n):
            frames.append(TF.to_tensor(Image.open(Path(self.frame_dir)/f'{i:05}.jpg')))
        label = float(self.label_file[idx+(self.n-1)].split()[0])  # but since we also get the next label, is as if we got the previous frame.
        sample = torch.cat(frames) if not self.transform else self.transform(torch.cat(frames))
        return [sample, torch.tensor((label, ))]

class OFDataset(Dataset):
    """Dataset for optical flow images"""
    def __init__(self, of_dir, label_f, transform=None):
        self.transform = transform
        self.len = len(list(Path(of_dir).glob('*.jpg')))
        self.of_dir = of_dir
        self.label_file = open(label_f).readlines()[1:]

    def __len__(self): 
        return self.len

    def __getitem__(self, idx):
        image = Image.open(Path(self.of_dir)/f'{idx:05}.jpg')
        of_tensor = TF.to_tensor(image)if not self.transform else self.transform(TF.to_tensor(image))
        label = float(self.label_file[idx].split()[0])
        return [of_tensor, torch.tensor((label, ))]

training_transform = transforms.Compose([transforms.RandomVerticalFlip(224)])

if "of" in args.dataset_mode:
    train_dataset = OFDataset(os.path.join(of_dir, "train"), os.path.join(labels_f, "train.txt"), transform=training_transform)
    test_dataset = OFDataset(os.path.join(of_dir, "test"), os.path.join(labels_f, "test.txt"))
    in_c = 3  # number of input channels
elif "d" in args.dataset_mode:
    train_dataset = NFrameDataset(os.path.join(of_dir, "train"), os.path.join(labels_f, "train.txt"), n=args.n, transform=training_transform)
    test_dataset = NFrameDataset(os.path.join(of_dir, "test"), os.path.join(labels_f, "test.txt"), n=args.n)
    in_c = 3*args.n  # number of input channels
else:
    print("The specified dataset does not exists")
    sys.exit()


trainloader = DataLoader(test_dataset, batch_size=5, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=5, shuffle=True)

dataloaders = {'train': trainloader, 'val': testloader}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

num_c = 1 # number of classes to predict
if 'efficientnet' in args.model:
    v = args.model_num     # model version
    model = EfficientNet.from_pretrained(f'efficientnet-b{v}', in_channels=in_c, num_classes=num_c)
    model_name = f"efficientnet-b{v}"
elif 'resnet' in args.model:
    model = models.resnet50(pretrained=True)
    if in_c != 3:  # for the dataset with frames as channels
        model.conv1 = nn.Conv2d(in_c, 64, 7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(2048, out_features=num_c) # assuming that the fc7 layer has 512 neurons, otherwise change it 
    model_name = "resnet50"
else:
    sys.exit("The model specified is not defined")

model.to(device)
params_to_update = model.parameters()
criterion = nn.MSELoss()
# Observe that all parameters are being optimized
if args.optimizer.upper() == "SGD":
    optimizer = optim.SGD(params_to_update, lr=args.lr, momentum=0.9, weight_decay=1e-4)
elif args.optimizer.lower() == "adam":
    optimizer = optim.Adam(params_to_update, lr=args.lr)
else:
    sys.exit("Optimizer not valid")
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150, 200, 300])
train_model(model, dataloaders, criterion, optimizer, num_epochs=args.epochs, scheduler=scheduler, save_path='./results', model_name=model_name)
