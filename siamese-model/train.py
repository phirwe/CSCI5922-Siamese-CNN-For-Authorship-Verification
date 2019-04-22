from torch.utils.data import DataLoader
from Dataset import AuthorsDataset
from torch import optim
import torchvision
import numpy as np
from Model import *
import matplotlib.pyplot as plt
import os

torch.cuda.set_device(0)
device = torch.device("cuda:0")

model = ResnetSiamese([1,1,1,1], 10)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

#model.cuda()

train_dataset = AuthorsDataset(
    root_dir='Dataset',
    path='train.txt',
)

train_loader = DataLoader(
    train_dataset,
    batch_size=20,
    shuffle=True
)

for epoch in range(20):

    for batch_idx,(X1,X2,Y) in enumerate(train_loader):

        #X1,X2,Y = X1.to(device),X2.to(device),Y.to(device)

        Y_hat = model.forward(X1,X2)
        loss = criterion(Y_hat, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print("EPOCH: %d\t BATCH: %d\tTRAIN LOSS = %f"%(epoch,batch_idx,loss.item()))

    if epoch%5 == 0:
        model_path = os.path.join('Model_Checkpoints',str(epoch))
        torch.save(model.state_dict(), model_path)
