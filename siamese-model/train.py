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
from datetime import datetime

# Parse command line flags
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--cuda", action="store_true")
parser.add_argument("-e", "--epochs", type=int, default=20)
parser.add_argument("--load_checkpoint", type=str, default=None)
args = parser.parse_args()

# Initialize model, loss and optimizer
model = ResnetSiamese([1,1,1,1], 10)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=1e-4)

if args.load_checkpoint:
    checkpoint_path = args.load_checkpoint
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer'])

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
    path='train.txt',
    transform=transforms.Compose([
        Pad((MAXWIDTH, MAXHEIGHT)),
        Threshold(177),
        ShiftAndCrop(700, random=True),
        Downsample(0.75),
    ]))

train_loader = DataLoader(
    train_dataset,
    batch_size=50,
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

    now = datetime.now()
    checkpoint_str = now.strftime("%m-%d-%Y_%H:%M:%S") + "_epoch" + str(epoch)
    checkpoint_path = os.path.join('Model_Checkpoints', checkpoint_str)
    checkpoint = {'state_dict': model.state_dict(),
                  'optimizer' : optimizer.state_dict()}

    torch.save(checkpoint, checkpoint_path)
