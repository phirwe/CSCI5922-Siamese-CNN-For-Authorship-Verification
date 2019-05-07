from torch.utils.data import DataLoader
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
parser.add_argument("data_path", type=str)
parser.add_argument("-c", "--cuda", action="store_true")
parser.add_argument("-e", "--epochs", type=int, default=50)
parser.add_argument("--load_checkpoint", type=str, default=None)
args = parser.parse_args()

# Initialize model, loss and optimizer
model = BaselineSiamese()
criterion = torch.nn.CrossEntropyLoss()
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
                            
    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2).type(torch.FloatTensor)
        power2 = torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2).type(torch.FloatTensor)
        power1 = torch.pow(euclidean_distance, 2).type(torch.FloatTensor)
        left = 1.0 - label
        right = label
        left = power1
        right = power2
        sum = left + right
        loss_contrastive = torch.mean(sum)
        #loss_contrastive = torch.mean((torch.tensor(1.0).to(device)-label) * torch.pow(euclidean_distance, 2) + (label) * power).type(torch.FloatTensor)
                                                    
                                                    
        return loss_contrastive
                                                        
                                                        
                                                        
criterion = ContrastiveLoss()
optimizer = optim.SGD(model.parameters(), lr = 2e-5)

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
    batch_size=50,
    shuffle=True
)

for epoch in range(args.epochs):
    batch_loss = []
    for batch_idx,(X1,X2,Y) in enumerate(train_loader):

        # Move batch to GPU
        if args.cuda:
            X1,X2,Y = X1.to(device),X2.to(device),Y.to(device)

        # Compute forward pass
        #Y_hat = model.forward(X1,X2)
        y1, y2 = model.forward(X1,X2)
        # Calculate training loss
        loss = criterion(y1.to(device),y2.to(device), Y)
        #l2_reg = torch.tensor(0.).to(device)
        #for temp in model.parameters():
        #    l2_reg += norm(temp)

        # Perform backprop and zero gradient
        # optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        batch_loss.append(loss.item())
        print("EPOCH: %d\t BATCH: %d\tTRAIN LOSS = %f"%(epoch,batch_idx,loss.item()))


    print (sum(batch_loss) / len(batch_loss))
    #:wq
    #for param in model.parameters():
        #print (param)
    # Save checkpoint
    now = datetime.now()
    checkpoint_str = "_epoch" + str(epoch)
    checkpoint_path = os.path.join('Model_Baseline_Checkpoints', checkpoint_str)
    checkpoint = {'state_dict': model.state_dict(),
                  'optimizer' : optimizer.state_dict()}

    torch.save(checkpoint, checkpoint_path)

