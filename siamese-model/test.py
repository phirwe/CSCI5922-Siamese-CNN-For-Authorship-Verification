from torch.utils.data import DataLoader
from Dataset import AuthorsDataset
import torchvision
import numpy as np
from Model import *
from Resnet import *
import matplotlib.pyplot as plt


data_path = './images'

train_dataset = AuthorsDataset(
    root_dir='Dataset',
    positive='positive.txt',
    negative='negative.txt'
)

train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    num_workers=1,
    shuffle=True
)

model = ResnetSiamese([1,1,1,1], 10, [100,2])
model2 = resnet18()

for batch_idx,(X1,X2,Y) in enumerate(train_loader):

    y = model.forward(X1,X2)
    print(y,Y)
