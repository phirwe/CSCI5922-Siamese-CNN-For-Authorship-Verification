#!/bin/bash -e

echo && echo Downloading Authors100 dataset && echo
curl https://transfer.sh/z8FLg/Authors.zip -o Authors.zip

echo && echo Extracting Authors100 dataset && echo
unzip Authors.zip
