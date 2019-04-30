from torch.utils.data import DataLoader
from Dataset import *
from torch import optim
from torchvision import transforms
import torchvision
import numpy as np
from Model import *
import matplotlib.pyplot as plt
import os
import sys
import argparse

# Initialize model, loss and optimizer
model = ResnetSiamese([1,1,1,1], 10)

# Parse command line flags
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--cuda", action="store_true")
parser.add_argument("-e", "--epochs", type=int, default=50)
parser.add_argument("--load_checkpoint", type=str, default=None)
args = parser.parse_args()

if args.load_checkpoint:
    checkpoint_path = args.load_checkpoint
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

# Initialize cuda
if args.cuda:
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    model.cuda()

# Constants from Authors100 dataset
MAXWIDTH = 2260
MAXHEIGHT = 337

valid_dataset = AuthorsDataset(
    root_dir='Dataset',
    path='valid.txt',
    transform=transforms.Compose([
        Pad((MAXWIDTH, MAXHEIGHT)),
        Threshold(177),
        CropWidth(700),
        Downsample(0.75)
    ]))

valid_loader = DataLoader(
    valid_dataset,
    batch_size=1,
    shuffle=True
)

acc = 0

for batch_idx,(X1,X2,Y) in enumerate(valid_loader):

    if args.cuda:
        X1,X2,Y = X1.to(device),X2.to(device),Y.to(device)

    Y_hat = model.forward(X1,X2)
    #print(Y_hat.argmax(),Y)
    if Y == Y_hat.argmax():
        acc += 1

print(acc/len(valid_dataset))
