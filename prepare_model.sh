function download_converter(){
    echo "Donwload & Convert ${1} ..."
    omz_downloader --name ${1}
    omz_converter --name ${1}
}

PRIMARY_MODEL="face-detection-adas-0001"
SECONDARY_MODEL="age-gender-recognition-retail-0013"

download_converter ${PRIMARY_MODEL}
download_converter ${SECONDARY_MODEL}