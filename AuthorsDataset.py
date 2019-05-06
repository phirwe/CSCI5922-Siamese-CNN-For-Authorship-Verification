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
import unittest

class AuthorsDataset(Dataset):
    """
    Implements the Authors100 dataset. Dataset is structured in pairs of images
    with a label 1 or 0 corresponding to whether the pair is by the same author
    or not respectively.

    Args:
        path (str):        path to file with listings of pairs/labels
        root_dir (str):    root directory of dataset
        transform (torchvision.transforms): transformations to be performed

    Returns:
        (tensor, tensor, bool):     processed pair of images and label for
                                    training/validation
    """
    def __init__(self, path, root_dir, transform=None):
        data_path = path
        self.dataframe = pd.read_csv(data_path,delimiter=' ',names=["filepath1","filepath2","label"])
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Get image paths and labels for a single pairing
        img1_path = os.path.join(self.root_dir, self.dataframe.iloc[idx,0])
        img2_path = os.path.join(self.root_dir, self.dataframe.iloc[idx,1])
        label = self.dataframe.iloc[idx,2]

        # Read images from dataset
        img1 = io.imread(img1_path)
        img2 = io.imread(img2_path)
        print (img1.shape)
        print (img2.shape)
        # Apply transformations
        if self.transform:
            (img1, img2) = self.transform((img1, img2))

        # Convert images to torch tensors
        img1 = torch.reshape(torch.from_numpy(img1),(1,img1.shape[0],img1.shape[1])).float()
        img2 = torch.reshape(torch.from_numpy(img2),(1,img2.shape[0],img2.shape[1])).float()

        return (img1, img2, label)


class Threshold(object):
    """
    Perform thresholding operation on handwriting sample to remove scanning
    artifacts. Sets all pixels above a certain threshold to white. Threshold
    value must be between 0 and 255

    Args:
        threshold_value (int):      pixel value cutoff for thresholding
        sample (ndarray, ndarray):  images to threshold

    Returns:
        (ndarray, ndarray):         thresholded images
    """
    def __init__(self, threshold_value):
        assert (0 < threshold_value < 255)
        self.threshold_value = threshold_value

    def __call__(self, sample):
        img1, img2 = sample
        img1[img1 > self.threshold_value] = 255
        img2[img2 > self.threshold_value] = 255

        return (img1, img2)

class Downsample(object):
    """
    Scale image down by a given factor. Scaling factor must be between 0 and 1

    Args:
        scale_factor (float):       scaling factor
        sample (ndarray, ndarray):  images to scale

    Returns:
        (ndarray, ndarray):         scaled images
    """
    def __init__(self, scale_factor):
        assert (0 < scale_factor < 1)
        self.scale_factor = scale_factor

    def __call__(self, sample):
        img1, img2 = sample
        img1 = transform.rescale(img1,self.scale_factor,multichannel=False)
        img2 = transform.rescale(img2,self.scale_factor,multichannel=False)

        return (img1, img2)

class Pad(object):
    """
    Pad image sample to maximum dimensions from Authors100 dataset. Horizontal
    padding is on the right, vertical padding is split between top and bottom

    Args:
        pad_size (int, int):        padded image dimensions
        sample (ndarray, ndarray):  images to pad

    Returns:
        (ndarray, ndarray):         padded images
    """
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
    """
    Horizontally crops an image to 'cropped_size' pixels wide. If 'random' is
    true, will start crop from random position, else image will be cropped from
    0 to 'cropped_size'

    Args:
        cropped_size (int):         width of cropped image
        random (bool):              flag for random cropping
        sample (ndarray, ndarray):  images to crop

    Returns:
        (ndarray, ndarray):         cropped images
    """
    def __init__(self, cropped_size, random=False):
        #self.assert...
        self.cropped_size = cropped_size
        self.bound = 1000 - cropped_size
        self.random = random

    def __call__(self, sample):
        img1, img2 = sample
        shift1, shift2 = 0,0
        if self.random:
            shift1 = random.randint(0,self.bound)
            shift2 = random.randint(0,self.bound)
        img1 = img1[:,shift1:shift1+self.cropped_size]
        img2 = img2[:,shift2:shift2+self.cropped_size]

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
