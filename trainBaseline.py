from torch.utils.data import DataLoader
<<<<<<< HEAD
import os
import sys
#sys.path.append('../')
from AuthorsDataset import *
from torch import optim
from torchvision import transforms
import torchvision
from torch import norm
import numpy as np
from Model import *
import matplotlib.pyplot as plt
import argparse
from datetime import datetime

# Parse command line flags
parser = argparse.ArgumentParser()
parser.add_argument("data_path", typr=str)
parser.add_argument("-c", "--cuda", action="store_true")
parser.add_argument("-e", "--epochs", type=int, default=20)
parser.add_argument("--load_checkpoint", type=str, default=None)
args = parser.parse_args()

# Initialize model, loss and optimizer
model = BaselineSiamese()
criterion = torch.nn.CrossEntropyLoss()
#optimizer = optim.SGD(model.parameters(), lr = 2e-5)
optimizer = optim.Adam(model.parameters())

if args.load_checkpoint:
    checkpoint_path = args.load_checkpoint
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

# Initialize cuda
if args.cuda:
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    model.cuda()

# Constants from Authors100 dataset
MAXWIDTH = 2260
MAXHEIGHT = 337

train_dataset = AuthorsDataset(
    root_dir='Dataset',
    path=args.data_path,
    transform=transforms.Compose([
        Pad((MAXWIDTH, MAXHEIGHT)),
        Threshold(177),
        ShiftAndCrop(700, random=True),
        Downsample(0.75),
    ]))

train_loader = DataLoader(
    train_dataset,
<<<<<<< HEAD
    batch_size=10,
=======
    batch_size=50,
>>>>>>> 0873334f314c023f538a3cc246d6de01d85e1cc6
    shuffle=True
)

file_ = open("loss_50000.txt", "a")

for epoch in range(args.epochs):
    batch_loss = []
    for batch_idx,(X1,X2,Y) in enumerate(train_loader):

        # Move batch to GPU
        
        # Move data to GPU
        if args.cuda:
            X1,X2,Y = X1.to(device),X2.to(device),Y.to(device)
            l2_reg = l2_reg.to(device)

        # Compute forward pass
        Y_hat = model.forward(X1,X2)


        # Perform backprop and zero gradient
        optimizer.zero_grad()
        # Apply Forward pass on input pairs
        Y_hat = model.forward(X1,X2)
        
        # Calculate loss on training data
        loss = criterion(Y_hat, Y)
       
        # Back propagate the loss and apply zero grad
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        batch_loss.append(loss.item())
        print("EPOCH: %d\t BATCH: %d\tTRAIN LOSS = %f"%(epoch,batch_idx,loss.item()))
    file_.write(str(loss.item())+"\n")        

    print (sum(batch_loss) / len(batch_loss))
    # Save model checkpoints
    if epoch%5 == 0:
        checkpoint_path = os.path.join('Model_Baseline_Checkpoints',"epoch" + str(epoch))
        checkpoint = {'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict()}

        torch.save(checkpoint, checkpoint_path)

