#!/bin/bash
source "$(dirname $(realpath $0))/utils.sh"

# Set the default value of the getopts variable 
RUN_CLI=false
MAGIC=true
SERVER=false
LOG="./docker/docker_info.log"

# Install pre-requirement
if [[ -z $(which jq) ]];then
    printd "Installing requirements .... " Cy
    sudo apt-get install jq -yqq
fi

# Help
function help(){
	echo "Run the iVIT-I environment."
	echo
	echo "Syntax: scriptTemplate [-g|wsmih]"
	echo "options:"
	echo "s		Server mode for non vision user."
	echo "c		Run as command line mode"
	echo "m		Print information with MAGIC."
	echo "h		help."
}

while getopts "g:wcsihmh" option; do
	case $option in
		s )
			SERVER=true ;;
		c )
			RUN_CLI=true ;;
		m )
			MAGIC=false ;;
		h )
			help; exit ;;
		\? )
			help; exit ;;
		* )
			help; exit ;;
	esac
done

# Setup Masgic package
if [[ ${MAGIC} = true ]];then
	if [[ -z $(which boxes) ]];then 
		printd "Preparing MAGIC "
		sudo apt-get install -qy boxes > /dev/null 2>&1; 
	fi
fi

# Setup variable
DOCKER_IMAGE="ov-aiot"
DOCKER_NAME="${DOCKER_IMAGE}"
MOUNT_CAMERA=""
WORKSPACE="/workspace"
SET_VISION=""

# Check if image come from docker hub
DOCKER_HUB_IMAGE="maxchanginnodisk/${DOCKER_IMAGE}"
if [[ ! $(check_image $DOCKER_HUB_IMAGE) -eq 0 ]];then
	DOCKER_IMAGE=${DOCKER_HUB_IMAGE}
	echo "From Docker Hub ... Update Docker Image Name: ${DOCKER_IMAGE}"
fi

# SERVER or DESKTOP MODE
if [[ ${SERVER} = false ]];then
	mode="DESKTOP"
	SET_VISION="-v /tmp/.x11-unix:/tmp/.x11-unix:rw -e DISPLAY=unix${DISPLAY}"
	# let display could connect by every device
	xhost + > /dev/null 2>&1
else
	mode="SERVER"
fi

# Combine Camera option
all_cam=$(ls /dev/video* 2>/dev/null)
cam_arr=(${all_cam})

for cam_node in "${cam_arr[@]}"
do
	MOUNT_CAMERA="${MOUNT_CAMERA} --device ${cam_node}:${cam_node}"
done

# Combine docker RUN_CMD line
DOCKER_CMD="docker run \
--name ${DOCKER_NAME} \
-it --rm \
--net=host --ipc=host \
-v /etc/localtime:/etc/localtime:ro \
--device /dev/dri \
--device-cgroup-rule='c 189:* rmw' \
-v /dev/bus/usb:/dev/bus/usb \
-w ${WORKSPACE} \
-v `pwd`:${WORKSPACE} \
${MOUNT_CAMERA} \
${SET_VISION} \
${DOCKER_IMAGE} \"bash\" \n"

# Show information
INFO="\n\
PROGRAMMER: Welcome to ${PROJECT} \n\
FRAMEWORK:  ${PLATFORM}\n\
MODE:  ${mode}\n\
DOCKER: ${DOCKER_IMAGE} \n\
CONTAINER: ${DOCKER_NAME} \n\
HOST: 0.0.0.0:${PORT} \n\
MOUNT_CAMERA:  $((${#cam_arr[@]}/2))\n\
COMMAND: bash \n"

# Print the INFO
print_magic "${INFO}" "${MAGIC}"
echo -e "Command: ${DOCKER_CMD}"

# Log
printf "$(date +%m-%d-%Y)" > "${LOG}"
printf "${INFO}" >> "${LOG}"
printf "\nDOCKER COMMAND: \n${DOCKER_CMD}" >> "${LOG}";

bash -c "${DOCKER_CMD}";
