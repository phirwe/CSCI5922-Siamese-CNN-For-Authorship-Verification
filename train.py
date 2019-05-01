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
from datetime import datetime

""" -----------------------SET HYPERPARAMETERS------------------------------ """
# Training Parameters
LEARNING_RATE   = 2e-5
BATCH_SIZE      = 50

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
parser.add_argument("-c", "--cuda", action="store_true")
parser.add_argument("-e", "--epochs", type=int, default=20)
parser.add_argument("--load_checkpoint", type=str, default=None)
args = parser.parse_args()

# Initialize model, loss and optimizer
model = ResnetSiamese(RESNET_LAYERS, RESNET_OUTSIZE)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=LEARNING_RATE)

# Load model checkpoint
if args.load_checkpoint:
    checkpoint_path = args.load_checkpoint
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])

# Initialize cuda
if args.cuda:
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    model.cuda()
    #optimizer.cuda()

# Constants from Authors100 dataset
MAXWIDTH = 2260
MAXHEIGHT = 337

train_dataset = AuthorsDataset(
    root_dir='Dataset',
    path=args.data_path,
    transform=transforms.Compose([
        Pad((MAXWIDTH, MAXHEIGHT)),
        Threshold(THRESHOLD_VALUE),
        ShiftAndCrop(CROP_SIZE, random=RANDOM_CROP),
        Downsample(DOWNSAMPLE_RATE),
    ]))

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

for epoch in range(args.epochs):

    for batch_idx,(X1,X2,Y) in enumerate(train_loader):

        # Move batch to GPU
        if args.cuda:
            X1,X2,Y = X1.to(device),X2.to(device),Y.to(device)

        # Compute forward pass
        Y_hat = model.forward(X1,X2)

        # Calculate training loss
        loss = criterion(Y_hat, Y)

        # Perform backprop and zero gradient
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print("EPOCH: %d\t BATCH: %d\tTRAIN LOSS = %f"%(epoch,batch_idx,loss.item()))

    # Save checkpoint
    now = datetime.now()
    checkpoint_str = now.strftime("%m-%d-%Y_%H:%M:%S") + "_epoch" + str(epoch)
    checkpoint_path = os.path.join('Model_Checkpoints', checkpoint_str)
    checkpoint = {'state_dict': model.state_dict(),
                  'optimizer' : optimizer.state_dict()}

    torch.save(checkpoint, checkpoint_path)
