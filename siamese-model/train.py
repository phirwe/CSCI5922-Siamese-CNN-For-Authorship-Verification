from torch.utils.data import DataLoader
from Dataset import AuthorsDataset
from torch import optim
import torchvision
import numpy as np
from Model import *
import matplotlib.pyplot as plt

torch.cuda.set_device(0)
device = torch.device("cuda:0")

model = ResnetSiamese([1,1,1,1], 10)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum=0.9)

#model.cuda()

train_dataset = AuthorsDataset(
    root_dir='Dataset',
    positive='positive.txt',
    negative='negative.txt'
)

train_loader = DataLoader(
    train_dataset,
    batch_size=20,
    shuffle=True
)

for epoch in range(100):

    for batch_idx,(X1,X2,Y) in enumerate(train_loader):

        #X1,X2,Y = X1.to(device),X2.to(device),Y.to(device)

        Y_hat = model.forward(X1,X2)
        loss = criterion(Y_hat, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print("EPOCH: %d\t BATCH: %d\tLOSS = %f"%(epoch,batch_idx,loss.item()))
