import torchvision
import torchvision.datasets as dset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
import pickle
from Dataset import AuthorsDataset
from torch import optim
import numpy as np
from Model import *
import matplotlib.pyplot as plt

class BaselineNet(nn.Module):

    def __init__(self, input_channels=1):
        super(BaselineNet, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=10, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=8, bias=False)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=4, bias=False)
        
        self.fc1 = nn.Linear(832896, 400)
        self.fc2 = nn.Linear(400, 200)
        self.dropout = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(200, 2)

    
    def forward(self, x):

        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.conv3(x)
        x = self.relu(x)
                
        x = x.view(x.size(0), -1)
                
        x = self.fc1(x)
        x = self.relu(x)
                
        x = self.fc2(x)
        x = self.relu(x)
        
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.relu(x)
        
        return x


class BaselineSiamese(nn.Module):

    def __init__(self, out_layers=2):
        super(BaselineSiamese, self).__init__()

        self.baselineNet = BaselineNet()
        self.fc1 = nn.Linear(8, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward_once(self, x):

        x = self.baselineNet(x)
        
        return x

    def forward(self, x, y):

        # Pass examples through siamese resnet
        f_x = self.forward_once(x)
        f_y = self.forward_once(y)

        # Concatenate outputs
        squared_diff = (f_x - f_y)**2
        hadamard = (f_x * f_y)
        x = torch.cat((f_x,f_y,squared_diff,hadamard),1)

        # Pass through fully connected layers
        x = self.fc1(x)

        x = self.softmax(x)

        return x
    

model = BaselineSiamese()
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)

train_dataset = AuthorsDataset(
    root_dir='Dataset',
    positive='positive.txt',
    negative='negative.txt'
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    num_workers=1,
    shuffle=True
)

for epoch in range(1):

    for batch_idx,(X1,X2,Y) in enumerate(train_loader):

        Y_hat = model.forward(X1,X2)
        
        loss = criterion(Y_hat, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print("EPOCH: %d\t BATCH: %d\tLOSS = %f"%(epoch,batch_idx,loss.item()))
