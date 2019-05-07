#!/bin/bash -e

echo && echo Creating GAN data && echo
python create_GAN_data.py 150 151 35

echo && echo Running GAN to generate forged handwriting && echo
python train.py --dataroot .pytorch-CycleGAN-and-pix2pix/datasets/150to151 --name 150to151_results --model cycle_gan
python test.py --dataroot .pytorch-CycleGAN-and-pix2pix/datasets/150to151 --name 150to151_results --model cycle_gan

echo && echo Generating GAN valid txt file && echo
python create_GAN_valid.py 150 151
