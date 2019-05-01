from torch.utils.data import DataLoader
from AuthorsDataset import *
from torch import optim
from torchvision import transforms
import torchvision
import numpy as np
from ResnetSiamese import *
import matplotlib.pyplot as plt
import os
import sys
import argparse

""" -----------------------SET HYPERPARAMETERS------------------------------ """
# Data Preprocessing
THRESHOLD_VALUE = 177
CROP_SIZE       = 700
RANDOM_CROP     = True
DOWNSAMPLE_RATE = 0.75

# Model Parameters NOTE: Modify at your own risk! Changes may be required to Model.py
RESNET_LAYERS   = [1,1,1,1]
RESNET_OUTSIZE  = 10
""" ------------------------------------------------------------------------ """

# Parse command line flags
parser = argparse.ArgumentParser()
parser.add_argument("data_path", type=str)
parser.add_argument("checkpoint", type=str)
parser.add_argument("-c", "--cuda", action="store_true")
parser.add_argument("-e", "--epochs", type=int, default=20)
args = parser.parse_args()

# Initialize model, loss and optimizer
model = ResnetSiamese(RESNET_LAYERS, RESNET_OUTSIZE)
criterion = torch.nn.CrossEntropyLoss()

# Load model checkpoint
if args.checkpoint:
    checkpoint_path = args.checkpoint
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
    path=args.data_path,
    transform=transforms.Compose([
        Pad((MAXWIDTH, MAXHEIGHT)),
        Threshold(THRESHOLD_VALUE),
        ShiftAndCrop(CROP_SIZE, random=RANDOM_CROP),
        Downsample(DOWNSAMPLE_RATE),
    ]))

valid_loader = DataLoader(
    valid_dataset,
    batch_size=10,
    shuffle=True
)

acc = 0
false_pos = 0
false_neg = 0

for batch_idx,(X1,X2,Y) in enumerate(valid_loader):

    if args.cuda:
        X1,X2,Y = X1.to(device),X2.to(device),Y.to(device)

    # Compute accuracy
    Y_hat = model.forward(X1,X2)
    _,Y_hat = Y_hat.max(1)
    acc += (Y_hat == Y).sum().item()

    false_pos += (Y_hat[Y_hat > Y]).sum().item()
    false_neg += (Y[Y_hat < Y]).sum().item()

print("-----------------------------")
print("VALIDATION ACCURACY: %f\t(%d of %d examples)"%(acc/len(valid_dataset),acc,len(valid_dataset)))
print("FALSE POSITIVE RATE: %f\t(%d of %d examples)"%(false_pos/len(valid_dataset),false_pos,len(valid_dataset)))
print("FALSE NEGATIVE RATE: %f\t(%d of %d examples)"%(false_neg/len(valid_dataset),false_neg,len(valid_dataset)))
print("-----------------------------")
