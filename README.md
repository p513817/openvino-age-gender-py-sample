# Face recognition with age and gender
The Intel OpenVINO Python Sample for Face Recognition and Age, Gender Recognition.

# How to work

* Build docker image and run container
    ```bash
    # Build image
    $ ./docker/build.sh

    # Run container
    $ ./docker/run.sh
    ```

* Download and Convert Model
    ```bash
    $ ./prepare_mdoel.sh

    $ ls intel/
    age-gender-recognition-retail-0013  face-detection-adas-0001
    ```

* Run demo
    ```bash
    python3 demo.py -c config.json

    # Run default : age and gender
    python3 demo.py -c config.json --no-app

    ```
    * Fetures
        * press a to toggle full screen.
        * press q to quit.
        * press c to change color.
    * Default
        ![demo](assest/demo-gender-age.png)
    * Smart Retail - [smart retail] recommand products for different age and gender.
        ![demo](assest/demo-smart-retail.png)

# Configuration

* About Model
    Define `primary` and `secondary` model parameters.
    ```JSON
    {
        "primary":{
            "path": "./intel/face-detection-adas-0001/FP16/face-detection-adas-0001",
            "device": "CPU",
            "thres": 0.6,
            "category":{
                "1": "face"
            }
        },
        "secondary":{
            "path": "./intel/age-gender-recognition-retail-0013/FP16/age-gender-recognition-retail-0013",
            "device": "MYRIAD",
            "thres": 0.6,
            "category": {
                "0": "female",
                "1": "male"
            }
        }
    }
    ```

* About Application
    ```JSON
    {
        "male": {
            "60-100": "Viagra 50% off, Patek Philippe 3% off, Callaway 20% off",
            "30-60": "Bose 10% off, Apple 3% off, FNAC 30% off, OSIM 15% off",
            "0-30": "Nike 20% off, Bandai 5% off, ASUS ROG 10% off"
        },
        "female": {
            "60-100": "Chanel 5% off, CTF 10% off, Bvlgari 10% off",
            "30-60": "Combi 20% off, SK-II 10% off, Lutein 30% off, Philps 10% off ",
            "0-30": "Dior/ YSL cosmetics 10% off, 10/10 perfums 20% off, Aesop 20% off"
        }
    }
    ```
