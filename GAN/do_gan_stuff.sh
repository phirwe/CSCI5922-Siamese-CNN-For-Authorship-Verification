#!/bin/bash -e

echo && echo Creating GAN data
python create_GAN_data.py 150 151 35
echo Created train-test files && echo

echo && echo Running GAN to generate forged handwriting && echo
python pytorch-CycleGAN-and-pix2pix/train.py --dataroot ./pytorch-CycleGAN-and-pix2pix/datasets/150to151 --name 150to151_results --model cycle_gan --preprocess none
python pytorch-CycleGAN-and-pix2pix/test.py --dataroot ./pytorch-CycleGAN-and-pix2pix/datasets/150to151 --name 150to151_results --model cycle_gan

echo && echo Generating GAN valid txt file && echo
python create_GAN_valid.py 150 151

echo && echo All GAN stuff is done, run the model through gan_images to get the details && echo
