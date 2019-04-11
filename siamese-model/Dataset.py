import os
import torch
import glob
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import math

# Constants from Authors100 dataset
MAXWIDTH = 2260
MAXHEIGHT = 337

### UTILITY FUNCTIONS FOR DATA PREPROCESSING ###

def pad(image):
    """Pad an image to the maximum image size in the dataset"""
    y,x = image.shape
    x_diff,y_diff = MAXWIDTH-x,MAXHEIGHT-y
    top,bottom = math.ceil(y_diff/2.),math.floor(y_diff/2.)
    right = x_diff
    return np.pad(image,((top,bottom),(0,right)),'constant',constant_values=255)

def threshold(image, threshold):
    image[image > threshold] = 255
    return image

def scale(image, scale):
    return transform.rescale(image,scale,multichannel=False)

def crop(image, cropped_size):
    return image[:,:cropped_size]

def show_image(image):
    """Display image for testing"""
    plt.imshow(image,cmap='gray')
    plt.show()

class AuthorsDataset(Dataset):

    def __init__(self, positive, negative, root_dir, scale=0.75, threshold=170, cropped_size=1000):
        pos_path = os.path.join(root_dir,positive)
        neg_path = os.path.join(root_dir,negative)
        self.positive_frame = pd.read_csv(pos_path,delimiter=' ',names=["filepath1","filepath2","label"])
        self.negative_frame = pd.read_csv(neg_path,delimiter=' ',names=["filepath1","filepath2","label"])
        self.dataframe = pd.concat([self.positive_frame, self.negative_frame])
        self.root_dir = root_dir
        self.scale = scale
        self.threshold = threshold
        self.cropped_size = cropped_size

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img1_path = os.path.join(self.root_dir, self.dataframe.iloc[idx,0])
        img2_path = os.path.join(self.root_dir, self.dataframe.iloc[idx,1])
        label = self.dataframe.iloc[idx,2]

        img1 = io.imread(img1_path)
        img2 = io.imread(img2_path)

        img1 = crop(scale(threshold(pad(img1), self.threshold), self.scale), self.cropped_size)
        img2 = crop(scale(threshold(pad(img2), self.threshold), self.scale), self.cropped_size)

        img1 = torch.reshape(torch.from_numpy(img1),(1,img1.shape[0],img1.shape[1])).float()
        img2 = torch.reshape(torch.from_numpy(img2),(1,img2.shape[0],img2.shape[1])).float()

        #label = torch.Tensor(label).long()

        return (img1, img2, label)
