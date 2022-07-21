import cv2, sys, os, time
import numpy as np
from openvino.inference_engine import IENetwork, IECore

model_name = 'face-detection-adas-0001'
path = "./intel/face-detection-adas-0001/FP16/"
ie = IECore()                                                   # Load CPU extenstion
net = ie.read_network(model=path+model_name+'.xml', 
                    weights=path+model_name+'.bin')             # Read IR model

input_blob = next(iter(net.input_info))                             # Get input name
out_blob = next(iter(net.outputs))                              # Get output name
print(out_blob)

batch, channel, height, width = net.input_info[input_blob].input_data.shape       # Get input shape
print(batch, channel, height, width)
# sys.exit()
print("Load IR to device")
exec_net = ie.load_network(network=net, device_name='CPU')      # Load IR model to device

print("Start")
cap = cv2.VideoCapture(0)
while True:
    ret, image = cap.read()
    cv2.imshow("input", image)                                      # Read image and manipulate
    ori_shape = image.shape                                         # Record original size
    image = cv2.resize(image, (width, height))                      # Resize to network image size
    t_image = image.transpose((2, 0, 1))                            # Change data layout from HWC to CHW

    """Infer!!!"""
    res = exec_net.infer(inputs={input_blob: t_image})              # Inference
    
    print(res['detection_out'].shape)
    idx = np.argsort(np.squeeze(res[out_blob][0]))[::-1]

    for i in range(res['detection_out'].shape[2]):                  #  res[0][0][index] -> [image_id, label, conf, x_min, y_min, x_max, y_max] 
        out = res['detection_out'][0][0][i]
        if out[2] > 0.5:
            x_l = int(width * out[3])
            y_l = int(height * out[4])
            x_u = int(width * out[5])
            y_u = int(height * out[6])
            cv2.rectangle(image, (x_l, y_l), (x_u, y_u), (0, 255, 0), 2)

    image = cv2.resize(image, (ori_shape[1], ori_shape[0]))
    
    cv2.imshow("output", image)  
    #cv2.imwrite('detect_face.jpg', image)
    print(res['detection_out'][0][0][0])
    if 0xFF & cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
