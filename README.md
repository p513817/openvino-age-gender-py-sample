# Face recognition with age and gender
The Intel OpenVINO Python Sample for Face Recognition and Age, Gender Recognition.

# How to work

* Build docker image and run container
    ```bash
    # Build image
    $ ./docker/build.sh

    # Run container
    $ ./docke/run.sh
    ```

* Download and Convert Model
    ```bash
    $ ./prepare_mdoel.sh

    $ ls intel/
    age-gender-recognition-retail-0013  face-detection-adas-0001
    ```

* Run demo
    ```bash
    # Run default : age and gender
    python3 demo.py -c config.json

    # With custom application
    python3 demo.py -c config.json --app
    ```
    * Fetures
        * press a to toggle full screen.
        * press q to quit.
        * press c to change color.
    * Default
        ![demo](assest/demo.png)
    * Application - [smart retail] recommand products for different age and gender.
        ![demo](assest/smart_retail.png)

# Configuration
Define `primary` and `secondary` model parameters.
```json
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

