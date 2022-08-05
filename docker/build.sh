#!/bin/bash
CONF="ivit-i.json"


# Store the utilities
ROOT=$(dirname `realpath $0`)
source "${ROOT}/utils.sh"

# Install pre-requirement
if [[ -z $(which jq) ]];then
    printd "Installing requirements .... " Cy
    sudo apt-get install jq -yqq
fi

# Concate name
IMAGE_NAME="ov-aiot"
printd "Concatenate docker image name: ${IMAGE_NAME}" Cy

# Build the docker image
cd $ROOT
printd "Build the docker image. (${IMAGE_NAME})" Cy
docker build -t ${IMAGE_NAME} .