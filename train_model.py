import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
from efficientnet_pytorch import EfficientNet
from trainer import train_model
from pathlib import Path
import numpy as np
import multiprocessing
import argparse
import os
from PIL import Image
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
parser.add_argument('--model_num', default=0, type=int,
                    help='number for efficientnet')

args = parser.parse_args()

# directory with the optical flow images
of_dir = os.path.join(args.dataset)
# labels as txt file
labels_f = os.path.join(args.labels)
class OFDataset(Dataset):

    def __init__(self, of_dir, label_f):
        self.len = len(list(Path(of_dir).glob('*.jpg')))
        self.of_dir = of_dir
        self.label_file = open(label_f).readlines()

    def __len__(self): 
        return self.len

    def __getitem__(self, idx):
        image = Image.open(Path(self.of_dir)/f'{idx:05}.jpg')
        of_tensor = TF.to_tensor(image)
        label = float(self.label_file[idx].split()[0])
        return [of_tensor, torch.tensor((label, ))]

train_dataset = OFDataset(os.path.join(of_dir, "train"), os.path.join(labels_f, "train.txt"))
test_dataset = OFDataset(os.path.join(of_dir, "test"), os.path.join(labels_f, "test.txt"))

trainloader = DataLoader(test_dataset, batch_size=5, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=5, shuffle=True)

dataloaders = {'train': trainloader, 'val': testloader}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

v = args.model_num     # model version
in_c = 3  # number of input channels
num_c = 1 # number of classes to predict

model = EfficientNet.from_pretrained(f'efficientnet-b{v}', in_channels=in_c, num_classes=num_c)
model.to(device)
params_to_update = model.parameters()
criterion = nn.MSELoss()
# Observe that all parameters are being optimized
if args.optimizer.upper() == "SGD":
    optimizer = optim.SGD(params_to_update, lr=args.lr, momentum=0.9, weight_decay=1e-4)
elif args.optimizer.lower() == "adam":
    optimizer = optim.Adam(params_to_update)
else:
    print("Optimizer not valid")
    sys.exit()
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150, 200, 300])
train_model(model, dataloaders, criterion, optimizer, num_epochs=args.epochs, scheduler=scheduler, save_path='./results', model_name=f"efficientnet-b{v}")
