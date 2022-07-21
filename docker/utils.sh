#!/bin/bash

REST='\e[0m';
GREEN='\e[0;32m';
BGREEN='\e[7;32m';
BRED='\e[7;31m';
Cyan='\033[0;36m';
BCyan='\033[7;36m'

function printd(){            
    
    if [ -z $2 ];then COLOR=$REST
    elif [ $2 = "G" ];then COLOR=$GREEN
    elif [ $2 = "R" ];then COLOR=$BRED
    elif [ $2 = "Cy" ];then COLOR=$Cyan
    elif [ $2 = "BCy" ];then COLOR=$BCyan
    else COLOR=$REST
    fi

    echo -e "$(date +"%T") ${COLOR}$1${REST}"
}

function check_image(){ 
	echo "$(docker images --format "table {{.Repository}}:{{.Tag}}" | grep ${1} | wc -l )" 
}
function check_container(){ 
	echo "$(docker ps -a --format "{{.Names}}" | grep ${1} | wc -l )" 
}

function check_container_run(){
	echo "$( docker container inspect -f '{{.State.Running}}' ${1} )"
}

function lower_case(){
	echo "$1" | tr '[:upper:]' '[:lower:]'
}
function upper_case(){
	echo "$1" | tr '[:lower:]' '[:upper:]'
}

function print_magic(){
	info=$1
	magic=$2
	echo ""
	if [[ $magic = true ]];then
		echo -e $info | boxes -d dog -s 80x10
	else
		echo -e $info
	fi
	echo ""
}