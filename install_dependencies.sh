#!/bin/bash -e

echo && echo Installing dependencies && echo
pip install -r requirements.txt

echo && echo Finished installing dependencies now calling download_data.sh && echo
./Dataset/download_data.sh
