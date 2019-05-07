import os
import sys

authorA = sys.argv[1]
authorB = sys.argv[2]
train_size = int(0.8 * int(sys.argv[3]))
test_size = int(sys.argv[3]) - train_size
datadir = '../Dataset/Authors/'

train_count = 0
test_count = 0
fileA = {}
fileB = {}

for x in os.listdir(datadir + authorA + '/'):
    if x != '.DS_Store':
        if 'train' not in fileA.keys():
            fileA['train'] = []
        if 'test' not in fileA.keys():
            fileA['test'] = []
        if len(fileA['train']) < 28:
            fileA['train'].append(x)
        elif len(fileA['test']) < 7:
            fileA['test'].append(x)
        else:
            break 
    
rootdir = './pytorch-CycleGAN-and-pix2pix/datasets/' + authorA + 'to' + authorB
trainA_dir = rootdir + '/trainA'
trainB_dir = rootdir + '/trainB'
testA_dir = rootdir + '/testA'
testB_dir = rootdir + '/testB'

# Comment these lines out if directories already exist
os.system('mkdir ' + rootdir)
os.system('mkdir ' + trainA_dir)
os.system('mkdir ' + trainB_dir)
os.system('mkdir ' + testA_dir)
os.system('mkdir ' + testB_dir)


for x in os.listdir(datadir + authorB + '/'):
    if x != '.DS_Store':
        if 'train' not in fileB.keys():
            fileB['train'] = []
        if 'test' not in fileB.keys():
            fileB['test'] = []
        if len(fileB['train']) < train_size:
            fileB['train'].append(x)
        elif len(fileB['test']) < test_size:
            fileB['test'].append(x)
        else:
            break 

for i in range(train_size):
    nameA = fileA['train'][i]
    nameB = fileB['train'][i]
    os.system('cp ' + datadir + authorA + '/' + nameA + ' ' + trainA_dir + '/' + nameA)
    os.system('cp ' + datadir + authorB + '/' + nameB + ' ' + trainB_dir + '/' + nameB)
    if i < test_size:
        testA = fileA['test'][i]
        testB = fileB['test'][i]
        os.system('cp ' + datadir + authorA + '/' + testA + ' ' + testA_dir + '/' + testA)
        os.system('cp ' + datadir + authorB + '/' + testB + ' ' + testB_dir + '/' + testB)
        
