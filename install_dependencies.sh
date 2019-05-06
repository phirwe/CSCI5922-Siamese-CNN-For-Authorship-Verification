#!/bin/bash -e

echo && echo Installing dependencies && echo
sudo pip install -r requirements.txt

echo && echo Finished installing dependencies now calling download_data.sh && echo
./Dataset/download_data.sh

echo && echo Finished downloading data now creating data for Cycle-GAN && echo
./GAN/make_data.sh
