import torchvision
import torchvision.datasets as dset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
import pickle
import os, sys
from AuthorsDataset import *
from torch import optim
import numpy as np
import matplotlib.pyplot as plt

class BaselineNet(nn.Module):

    """
    Baseline Siamese Network described in the paper - http://cs231n.stanford.edu/reports/2017/pdfs/801.pdf
    
    Architecture Layout:
        Conv2D - 32 filters
        ReLU
        MaxPool 2x2
        Conv2D - 64 filters
        ReLU
        MaxPool 2x2
        Conv2D - 64 filters
        ReLU
        Fully Connected
        Dropout
        Fully Connected
    """
    def __init__(self, input_channels=1):
        super(BaselineNet, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=10, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=8, bias=False)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=4, bias=False)
        
        self.fc1 = nn.Linear(421632, 400)
        self.fc2 = nn.Linear(400, 200)
        self.dropout = nn.Dropout(p=0.5)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
    
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
        
        x = self.dropout(x)
                
        x = self.fc2(x)
        x = self.relu(x)
        
        
        return x


class BaselineSiamese(nn.Module):
    
    """
    Implements Baseline siamese network which has few layers and large kernels. 
    Encodings of the image pairs are concatenated, then passed through a fully connected layer 
    and a softmax operation is applied.
    """
    
    def __init__(self, out_layers=2):
        super(BaselineSiamese, self).__init__()

        self.baselineNet = BaselineNet()
        self.fc = nn.Linear(200, out_layers)
        self.softmax = nn.Softmax(dim=1)

    def forward_once(self, x):

        x = self.baselineNet(x)
        return x

    def forward(self, x, y):

        # Pass examples through siamese resnet
        f_x = self.forward_once(x)
        f_y = self.forward_once(y)
        #return f_x, f_y
        # Concatenate outputs
        squared_diff = (f_x - f_y)**2
        hadamard = (f_x * f_y)
        x = torch.cat((f_x,f_y,squared_diff,hadamard),1)

        # Pass through fully connected layers
        x = self.fc(x)
        x = self.softmax(x)


        return x
