import cv2, sys
import numpy as np
from openvino.inference_engine import IENetwork, IECore

model_name = 'age-gender-recognition-retail-0013'
path = "./intel/age-gender-recognition-retail-0013/FP16/"
classes = {
    0: "Girl",
    1: "Boy"
}

ie = IECore()                                                   # Load CPU extenstion
net = ie.read_network(model=path+model_name+'.xml', 
                    weights=path+model_name+'.bin')             # Read IR model

input_blob = next(iter(net.input_info))                             # Get input name

out_blob = []
for output in iter(net.outputs):
    out_blob.append(output)

batch, channel, height, width = net.input_info[input_blob].input_data.shape       # Get input shape

print("Load IR to device")
exec_net = ie.load_network(network=net, device_name='CPU')      # Load IR model to device

print("Start")
cap = cv2.VideoCapture(0)
while True:
    ret, ori_image = cap.read()
    # cv2.imshow("input", image)                                      # Read image and manipulate
    ori_shape = ori_image.shape                                         # Record original size
    image = cv2.resize(ori_image, (width, height))                      # Resize to network image size
    t_image = image.transpose((2, 0, 1))                            # Change data layout from HWC to CHW

    """Infer!!!"""
    res = exec_net.infer(inputs={input_blob: t_image})              # Inference
    
    for blob in out_blob:

        if blob == "age_conv3":
            age_info = "Age: {}".format(res[blob][0][0][0][0]*100)
            print(age_info)
        else:
            prob = res[blob][0]
            gender_info = "Gender: {}".format(classes[np.argmax(prob)])
            print(gender_info)
    
    cv2.putText(ori_image, age_info+"\t"+gender_info, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow("results", ori_image)  
    
    if 0xFF & cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
