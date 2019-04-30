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
import random

class AuthorsDataset(Dataset):

    def __init__(self, path, root_dir, transform=None):
        data_path = os.path.join(root_dir,path)
        self.dataframe = pd.read_csv(data_path,delimiter=' ',names=["filepath1","filepath2","label"])
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img1_path = os.path.join(self.root_dir, self.dataframe.iloc[idx,0])
        img2_path = os.path.join(self.root_dir, self.dataframe.iloc[idx,1])
        label = self.dataframe.iloc[idx,2]

        img1 = io.imread(img1_path)
        img2 = io.imread(img2_path)

        if self.transform:
            (img1, img2) = self.transform((img1, img2))

        #show_image(img1,img2)

        img1 = torch.reshape(torch.from_numpy(img1),(1,img1.shape[0],img1.shape[1])).float()
        img2 = torch.reshape(torch.from_numpy(img2),(1,img2.shape[0],img2.shape[1])).float()

        return (img1, img2, label)


class Threshold(object):

    def __init__(self, threshold_value):
        #assertTrue(0 < threshold_value < 255)
        self.threshold_value = threshold_value

    def __call__(self, sample):
        img1, img2 = sample
        img1[img1 > self.threshold_value] = 255
        img2[img2 > self.threshold_value] = 255

        return (img1, img2)

class Downsample(object):

    def __init__(self, scale_factor):
        #assertTrue(0 < scale_factor < 1)
        self.scale_factor = scale_factor

    def __call__(self, sample):
        img1, img2 = sample
        img1 = transform.rescale(img1,self.scale_factor,multichannel=False)
        img2 = transform.rescale(img2,self.scale_factor,multichannel=False)

        return (img1, img2)

class Pad(object):

    def __init__(self, pad_size):
        #self.assert
        self.pad_x = pad_size[0]
        self.pad_y = pad_size[1]

    def __call__(self, sample):
        img1, img2 = sample

        y,x = img1.shape
        x_diff,y_diff = self.pad_x-x,self.pad_y-y
        top,bottom = math.ceil(y_diff/2.),math.floor(y_diff/2.)
        right = x_diff
        img1 = np.pad(img1,((top,bottom),(0,right)),'constant',constant_values=255)

        y,x = img2.shape
        x_diff,y_diff = self.pad_x-x,self.pad_y-y
        top,bottom = math.ceil(y_diff/2.),math.floor(y_diff/2.)
        right = x_diff
        img2 = np.pad(img2,((top,bottom),(0,right)),'constant',constant_values=255)

        return (img1, img2)

class ShiftAndCrop(object):

    def __init__(self, cropped_size):
        #self.assert...
        self.cropped_size = cropped_size
        self.bound = 1000 - cropped_size

    def __call__(self, sample):
        img1, img2 = sample
        rand1 = random.randint(0,self.bound)
        rand2 = random.randint(0,self.bound)
        img1 = img1[:,rand1:rand1+self.cropped_size]
        img2 = img2[:,rand2:rand2+self.cropped_size]

        return (img1, img2)


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

def show_image(img1, img2):
    """Display image for testing"""
    plt.subplot(2,1,1)
    plt.imshow(img1,cmap='gray')
    plt.subplot(2,1,2)
    plt.imshow(img2,cmap='gray')
    plt.show()
