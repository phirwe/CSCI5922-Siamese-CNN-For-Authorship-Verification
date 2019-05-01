import os
from itertools import islice
import shutil
import glob
import random

"""

Utility script for randomly creating positive and negative pairs of images
from the Authors100 dataset. Script will generate an equal number
(DATASET_SIZE of positive and negative examples, and put them in training
and validation files according to TRAIN_TEST_SPLIT

"""

DATASET_SIZE = 10000
TRAIN_TEST_SPLIT = 0.95

dataset_path = "./Authors"

train = open("train_" + str(DATASET_SIZE)+ ".txt","w")
valid = open("valid_" + str(DATASET_SIZE)+ ".txt","w")

author_paths = glob.glob(os.path.join(dataset_path,'*'))

author_ids = []
for i in range(len(author_paths)):
    author_ids.append(author_paths[i][-3:])

train_size = int(DATASET_SIZE * TRAIN_TEST_SPLIT)
valid_size = DATASET_SIZE - train_size
idx1,idx2 = 0,0
pos_id = 0
neg_ids = (0,0)
poslabel = 1
neglabel = 0

# Generate training and valid sets
for i in range(DATASET_SIZE):

    # Randomly get author IDs for positive and negative pairs
    pos_id = author_paths[random.randint(0,99)]
    neg_ids = (author_paths[random.randint(0,99)],author_paths[random.randint(0,99)])

    # Ensure different author ids for negative pair
    while neg_ids[0] == neg_ids[1]:
        neg_ids = (author_paths[random.randint(0,99)],author_paths[random.randint(0,99)])

    # Randomly get writing samples from each author id
    pos_filepaths = glob.glob(os.path.join(pos_id,"*.png"))
    idx1,idx2 = random.randint(0,len(pos_filepaths)-1),random.randint(0,len(pos_filepaths)-1)

    # Ensure positive writing samples are not the same sample
    while idx1 == idx2:
        idx1,idx2 = random.randint(0,len(pos_filepaths)-1),random.randint(0,len(pos_filepaths)-1)

    # Write positive example
    if i < train_size:
        train.write(pos_filepaths[idx1]+' '+pos_filepaths[idx2]+' '+str(poslabel)+'\n')
    else:
        valid.write(pos_filepaths[idx1]+' '+pos_filepaths[idx2]+' '+str(poslabel)+'\n')

    neg_filepaths = (glob.glob(os.path.join(neg_ids[0],"*.png")),glob.glob(os.path.join(neg_ids[1],"*.png")))
    idx1,idx2 = random.randint(0,len(neg_filepaths[0])-1),random.randint(0,len(neg_filepaths[1])-1)

    # Write negative example
    if i < train_size:
        train.write(neg_filepaths[0][idx1]+' '+neg_filepaths[1][idx2]+' '+str(neglabel)+'\n')
    else:
        valid.write(neg_filepaths[0][idx1]+' '+neg_filepaths[1][idx2]+' '+str(neglabel)+'\n')

train.close()
valid.close()
