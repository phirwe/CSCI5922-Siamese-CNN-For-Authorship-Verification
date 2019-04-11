from torch.utils.data import DataLoader
from Dataset import AuthorsDataset
from torch import optim
import torchvision
import numpy as np
from Model import *
import matplotlib.pyplot as plt

model = ResnetSiamese([1,1,1,1], 10)
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

for epoch in range(100):

    for batch_idx,(X1,X2,Y) in enumerate(train_loader):

        Y_hat = model.forward(X1,X2)
        loss = criterion(Y_hat, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print("EPOCH: %d\t BATCH: %d\tLOSS = %f"%(epoch,batch_idx,loss.item()))
