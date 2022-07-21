from distutils.sysconfig import EXEC_PREFIX
import cv2, sys, os, time, logging
import numpy as np
from openvino.inference_engine import IENetwork, IECore


def preprocess(image, size):
    """ 
    1. Resize to network image size
    2. Change data layout from HWC to CHW
    """
    return cv2.resize(image, size).transpose((2, 0, 1))    

CV_WIN          = "Face Age Gender"
FONT_FACE       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE      = 0.5
FONT_THICK      = 1
FULL_SCREEN     = False

classes = {
    0: "Girl",
    1: "Boy"
}

ie = IECore() 

pri_path = "./intel/face-detection-adas-0001/FP16/face-detection-adas-0001"
sec_path = "./intel/age-gender-recognition-retail-0013/FP16/age-gender-recognition-retail-0013"

pri_net = ie.read_network(
    model = pri_path+'.xml',
    weights = pri_path+'.bin'
)

sec_net = ie.read_network(
    model = sec_path+'.xml',
    weights = sec_path+'.bin'
)

# Primary Model Info
pri_input_blob = next(iter(pri_net.input_info))                             # Get input name
pri_out_blob = next(iter(pri_net.outputs))                              # Get output name
pri_batch, pri_channel, pri_height, pri_width = pri_net.input_info[pri_input_blob].input_data.shape       # Get input shape

# print(pri_input_blob, pri_out_blob)
# print(pri_batch, pri_channel, pri_height, pri_width)

# Secondary Model Info
sec_input_blob = next(iter(sec_net.input_info))                             # Get input name
sec_out_blob = [ output for output in iter(sec_net.outputs) ]
sec_batch, sec_channel, sec_height, sec_width = sec_net.input_info[sec_input_blob].input_data.shape       # Get input shape

# Load IR model to device
pri_exec_net = ie.load_network(network=pri_net, device_name='CPU')      
sec_exec_net = ie.load_network(network=sec_net, device_name='CPU')      

print("Start")
cv2.namedWindow(CV_WIN, cv2.WND_PROP_FULLSCREEN)
cap = cv2.VideoCapture(0)
while True:

    # Get Input Data
    ret, image = cap.read()
    org_image = image.copy()
    org_height, org_width, _ = image.shape
    org_shape = image.shape

    # Pre-process
    temp_image = preprocess(image, (pri_width, pri_height) )

    # Inference
    res = pri_exec_net.infer(inputs={pri_input_blob: temp_image})
    
    # Parse Bounding Box
    for i in range(res[pri_out_blob].shape[2]):
        
        # res[0][0][index] -> [image_id, label, conf, x_min, y_min, x_max, y_max] 
        out = res[pri_out_blob][0][0][i]
        
        if out[2] > 0.5:
            x_l = int(org_width * out[3])
            y_l = int(org_height * out[4])
            x_u = int(org_width * out[5])
            y_u = int(org_height * out[6])

            x_l = x_l if x_l > 0 else 0
            y_l = y_l if y_l > 0 else 0
            x_u = x_u if x_u < org_shape[1] else org_shape[1]
            y_u = y_u if y_u < org_shape[0] else org_shape[0]

            # Get Face Area
            face_image = org_image[y_l:y_u, x_l:x_u]
            # Pre-process
            temp_image = preprocess(face_image, (sec_width, sec_height) )
            # Inference
            res2 = sec_exec_net.infer(inputs={sec_input_blob: temp_image})
    
            age_info, gender_info = None, None
            for blob in sec_out_blob:
                if blob == "age_conv3":
                    age_info = int(res2[blob][0][0][0][0]*100)
                else:
                    prob = res2[blob][0]
                    gender_info = classes[np.argmax(prob)]

            cv2.rectangle(org_image, (x_l, y_l), (x_u, y_u), (0, 255, 0), 2)

            text = f"{age_info}, {gender_info}"
            retval, baseLine = cv2.getTextSize(text, FONT_FACE, FONT_SCALE, FONT_THICK)
            text_x = x_l
            text_y = y_l-baseLine
            cv2.putText(org_image, text, (text_x, text_y), FONT_FACE, FONT_SCALE, (0, 255, 255), FONT_THICK, cv2.LINE_AA)

    org_image = cv2.resize(org_image, (org_shape[1], org_shape[0]))
    
    cv2.imshow(CV_WIN, org_image)  
    
    key = cv2.waitKey(1) & 0xFF
    if key in [ ord('q'), 27 ]:
        print("End")
        break
    if key in [ ord('a'), 1, 123 ]:
        
        FULL_SCREEN = not FULL_SCREEN
        cv2.setWindowProperty(CV_WIN,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN if FULL_SCREEN else cv2.WINDOW_NORMAL )
        
cv2.destroyAllWindows()
