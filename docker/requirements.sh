#!/bin/bash
source utils.sh

# Initial
printd "Initialize ... " Cy
apt-get update -qqy
apt-get install -qy figlet boxes tree > /dev/null 2>&1

ROOT=`pwd`
echo "Workspace is ${ROOT}" | boxes -p a1

printd "Install OpenCV " Cy
apt-get install -qqy libxrender1 libsm6 libxext6 #> /dev/null 2>&1

printd "Install other msicellaneous packages " Cy
apt-get -qy install bsdmainutils zip jq wget
pip3 install --disable-pip-version-check cython wget colorlog psutil

printd -e "Done${REST}"
