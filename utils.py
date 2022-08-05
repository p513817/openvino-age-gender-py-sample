# -*- coding: utf-8 -*-
import argparse, json
import cv2, sys, os, time, logging, colorlog, random, json
import numpy as np
from openvino.inference_engine import IENetwork, IECore

# Define OpenCV
CV_WIN          = "Face Age Gender"
FONT_FACE       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE      = 1
FONT_THICK      = 2
FULL_SCREEN     = True
LOGO_SCALE      = 0.15
INFO_SCALE      = 1.5

( TITLE_WIDTH, TEXT_HEIGHT), BASE_LINE = cv2.getTextSize(CV_WIN, FONT_FACE, FONT_SCALE, FONT_THICK)
TEXT_PADDING    = int(TEXT_HEIGHT * 0.8)

# Define Key
PRI             = "PRI"
SEC             = "SEC"
ORG             = "ORG"

# PATH            = "PATH"
MODEL_PATH      = "MODEL_PATH"
WEIGHT_PATH     = "WEIGHT_PATH"
NET             = "NET"
EXEC            = "EXEC"
IN_BLOB         = "INPUT_BLOB"
OUT_BLOB        = "OUT_BLOB"
RESULT          = "RESULT"
DEVICE          = "DEVICE"
BATCH           = "BATCH"
CHANNEL         = "CHANNEL"
HEIGHT          = "HEIGHT"
WIDTH           = "WIDTH"

def config_logging():
    logger = logging.getLogger()            # get logger
    logger.setLevel(logging.DEBUG)       # set level
    
    if not logger.hasHandlers():    # if the logger is not setup
        basic_formatter = logging.Formatter( "%(asctime)s [%(levelname)s] %(message)s (%(filename)s:%(lineno)s)", "%y-%m-%d %H:%M:%S")
        formatter = colorlog.ColoredFormatter( "%(asctime)s %(log_color)s [%(levelname)-.4s] %(reset)s %(message)s %(purple)s (%(filename)s:%(lineno)s)", "%y-%m-%d %H:%M:%S")
        # add stream handler
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(logging.DEBUG)
        logger.addHandler(stream_handler)

def get_cpu_temp():
    avg = 0
    try:
        import psutil
        KEY  = "coretemp"
        res  = psutil.sensors_temperatures() 
        temp = [ float(core.current) for core in res[KEY] ]
        avg  = round( sum(temp)/len(temp), 2)
    except Exception as e:
        logging.error("Get temperature error")
        avg = 0
    return avg

def preprocess(image, size):
    """ 
    1. Resize to network image size
    2. Change data layout from HWC to CHW
    """
    return cv2.resize(image, size).transpose((2, 0, 1))    

def add_logo(frame, logo):

    # get logo size
    height, width, channels = logo.shape

    # get binary mask which could parse out frame's pixel
    gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(gray, thresh=127, maxval=255, type=cv2.THRESH_BINARY)
    
    # this could parse out logo's pixel
    mask_inv = cv2.bitwise_not( mask)
    
    # get the background and frontground and add together
    roi = frame[0:height, 0:width]
    logo_bg = cv2.bitwise_and(roi, roi, mask = mask)
    logo_fg = cv2.bitwise_and(logo, logo, mask = mask_inv)
    dst = cv2.add( logo_bg, logo_fg)

    # add into frame
    frame[0:height, 0:width] = dst
    return frame

def bbox_scaler(bbox, trg_height, trg_width):

    x_l = int(trg_width * bbox[0])
    y_l = int(trg_height * bbox[1])
    x_u = int(trg_width * bbox[2])
    y_u = int(trg_height * bbox[3])

    x_l = x_l if x_l > 0 else 0
    y_l = y_l if y_l > 0 else 0
    x_u = x_u if x_u < trg_width else trg_width
    y_u = y_u if y_u < trg_height else trg_height
    
    return x_l, y_l, x_u, y_u

def get_random_color( format='bgr'):
    r = random.randint(0,255)
    g = random.randint(0,255)
    b = random.randint(0,255)
    return [r, g, b] if format=='rgb' else [b, g, r]

def capture_temperature(exec_net):
    try:
        return round(float(exec_net.get_metric("DEVICE_THERMAL")), 3)
    except Exception as e:
        return "Not support Temperature"

def capture_device_name(ie, device):
    try: 
        return ie.get_metric(metric_name="FULL_DEVICE_NAME", device_name=device)
    except Exception as e: 
        return ""

def draw_device_info(org_image, ie, net, start_pos=None, title="Device Information"):
    h,w,c = org_image.shape
    padding = 10
    # Combine data
    info = [ title ]
    info.append(" - {}: {}".format("Name", capture_device_name(ie, net[DEVICE])))
    info.append(" - {}: {}".format("Temp", capture_temperature(net[EXEC])))
    
    # Draw
    retval = cv2.getTextSize(title, FONT_FACE, FONT_SCALE, FONT_THICK)[0][1] + padding

    if start_pos==None:
        start_x, start_y = ( padding, (h-padding)-(retval*len(info)) )
    else:
        start_x, start_y = ( start_pos[0], (start_pos[1]-padding)-(retval*len(info)) )

    # Get First Position
    text_x, text_y = start_x, start_y
    
    for i in range(len(info)):
        text_x, text_y = text_x, text_y + (retval)
        cv2.putText(org_image, info[i], (text_x, text_y), FONT_FACE, FONT_SCALE, (0, 0, 0), FONT_THICK+1, cv2.LINE_AA)
        cv2.putText(org_image, info[i], (text_x, text_y), FONT_FACE, FONT_SCALE, (0, 0, 255), FONT_THICK, cv2.LINE_AA)
        
    return org_image, (start_x, start_y)

def read_json(path:str) -> dict:
    """ Read json file """
    with open(path, 'r') as f:
        conf = json.load(f)
    return conf

class DiscountInfo():

    def __init__(self, app_config="") -> None:

        self.palette = dict()
        self.informations = list()
        self.mapping_table = dict()
        self.age_range = dict()

        self.define_mapping_table(app_config)
        self.generate_palette()

    def define_mapping_table(self, app_config):

        logging.info("Parsing application config file: {}".format(app_config))
        self.mapping_table = read_json(app_config)
        self.get_age_range(self.mapping_table)

    def get_age_range(self, app_config:dict) -> None:
        """ 
        Get all age in config 
        """
        logging.info("Application Config")
        for gender, info in app_config.items():
            logging.info("  - {}".format(gender))

            self.age_range.update( { gender: list() } )

            for age_range, product in info.items():
                logging.info("      - {:<10}: {}".format(age_range, product))

                for age in age_range.split("-"):
                    age = int(age)
                    if not (age in self.age_range[gender]):
                        self.age_range[gender].append( age ) 
        
            self.age_range[gender].sort()

    def generate_palette(self):
        # Generate palette
        for gender, whole_info in self.mapping_table.items():
            for range, info in whole_info.items():
                self.informations.append(info)
                random_color = get_random_color()
                self.palette.update( { info: random_color })

        return self.palette

    def get_palette(self):
        return self.palette

    def get_info_color(self, info):
        if not (info in self.informations):
            raise Exception("{} not in mapping table".format(info))
        return self.palette[info]

    def formulation(self, gender, age):
        
        age, age_min, age_max = int(age), None, None

        for age_range in self.age_range[gender]:
            
            if age > age_range: 
                age_min = age_range
            
            if age <= age_range: 
                age_max = age_range

            if not None in [ age_min, age_max ]:
                break

        return "{}-{}".format(age_min, age_max)
    
    def __call__(self, age, gender):
        """ Call the DiscountInfo object
        Argument
            - age
                - type: int
            - gender
                - type: string
        Return
            - product
                - type: String
                - example: "20% Off for Rolex Watches"
        """
        return self.mapping_table[gender][self.formulation(gender, age)]

def parse_face_result(out):
    # -> [image_id, label, conf, x_min, y_min, x_max, y_max] 
    return out[:3], out[3:]
    
def parse_gender_age(dets, classes):
    age, gender = None, None
    for blob in dets:
        if blob == "age_conv3":
            age = int(dets[blob][0][0][0][0]*100)
            
        else:
            prob = dets[blob][0]
            gender = classes[str(np.argmax(prob))]
    
    return gender, age

def load_multi_network(conf):
    # Define nets
    nets = {
        PRI: dict(),
        SEC: dict(),
        ORG: dict()
    }

    # Load CPU extenstion
    ie = IECore()

    # Read IR model
    for TRG, path, device in zip([ PRI, SEC ], [ conf["primary"]["path"], conf["secondary"]["path"] ], [ conf["primary"]["device"], conf["secondary"]["device"] ]):

        nets[TRG][MODEL_PATH] = "{}.xml".format(path)
        nets[TRG][WEIGHT_PATH] = "{}.bin".format(path)
        nets[TRG][DEVICE] = device

        (   nets[TRG][NET], 
            nets[TRG][IN_BLOB], 
            nets[TRG][OUT_BLOB], 
            ( _, _, nets[TRG][HEIGHT], nets[TRG][WIDTH] ), 
            nets[TRG][EXEC] ) = load_network(
                                    ie = ie,
                                    model_path = nets[TRG][MODEL_PATH],
                                    weight_path = nets[TRG][WEIGHT_PATH],
                                    device = nets[TRG][DEVICE]
                                )
    logging.info("Load All Network Sccessfully")
    
    return nets

def load_network(ie, model_path, weight_path, device="CPU"):
    """
    Load IR Model and Parse Information.
    """
    logging.info("---")
    logging.info("Load Network")
    t_start = time.time()
    [ logging.info("\t- {}: {}".format(key, value)) for key, value in { "Model Path":model_path, "Weight Path":weight_path}.items() ]
    net = ie.read_network(
        model = model_path,
        weights = weight_path
    )
    
    input_blob = [ i_layer for i_layer in iter(net.input_info) ]
    output_blob = [ o_layer for o_layer in iter(net.outputs) ]
    batch, channel, height, width = net.input_info[input_blob[0]].input_data.shape
    
    logging.info("\t- Input Layer ({})".format( len(input_blob) ))
    [ logging.info("\t\t- Name: {}".format( layer )) for layer in input_blob ]

    logging.info("\t- Output Layer ({})".format( len(output_blob) ))
    [ logging.info("\t\t- Name: {}".format( layer )) for layer in output_blob ]
    
    logging.info("\t- Get Input Information")
    logging.info("\t\t- Input Shape: {}".format((batch, channel, height, width)))

    exec_net = ie.load_network(
        network = net, 
        device_name = device
    )
    t_end = time.time()
    logging.info("Loading Network Cost {:.3}s".format(t_end-t_start))

    return net, input_blob, output_blob ,(batch, channel, height, width), exec_net

def get_logo(path, img_size, scale):

    logo                = cv2.imread(path)
    (img_h, img_w)      = img_size
    (logo_h, logo_w)    = logo.shape[:2]

    trg_scale           = (img_w * scale / logo_w )
    (logo_h, logo_w )   = int(logo_h * trg_scale), int(logo_w * trg_scale)

    logo                = cv2.resize(logo, (logo_w, logo_h) )
    return logo

def define_cv_window(title, fullscreen=True, first_frame=[]):
    cv2.namedWindow(title, cv2.WND_PROP_FULLSCREEN)
    if fullscreen:
        cv2.setWindowProperty(CV_WIN,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN if FULL_SCREEN else cv2.WINDOW_NORMAL )

    if first_frame != []:
        cv2.imshow(CV_WIN, first_frame)

def put_text_border(frame, text, pos, fg=(0,0,255), bg=(0,0,0), border=1, scale=None):
    (x, y) = pos
    scale = FONT_SCALE if scale==None else scale
    cv2.putText(frame, text, pos, FONT_FACE, scale, bg, FONT_THICK+border, cv2.LINE_AA)
    cv2.putText(frame, text, pos, FONT_FACE, scale, fg, FONT_THICK, cv2.LINE_AA)
    return frame

def draw_gender_age(frame, xyxy, detections, org_size, trg_color):
    
    # Parse Information
    x1, y1, x2, y2  = xyxy
    org_h, org_w    = org_size

    # Get Text Size
    total_height    = BASE_LINE + ( (TEXT_HEIGHT + TEXT_PADDING)*len(detections))
    bottom =  (y1 - total_height if ( y2 + total_height >= org_h ) else y2 + BASE_LINE ) 

    # Run each information
    for idx, text in enumerate(detections):

        (text_width, text_height), base_line = cv2.getTextSize(text, FONT_FACE, FONT_SCALE, FONT_THICK)
    
        text_x = x1
        text_y = bottom + (text_height + base_line)*(idx+1)
        
        lt = (text_x, text_y - text_height - int(base_line*0.5))
        rb = (text_x + text_width, text_y)

        put_text_border(frame, text, (text_x, text_y), trg_color)
    return frame

def draw_information(frame, informations:dict, font_scale = None):
    
    # Get frame size
    height, width = frame.shape[:2]

    # Update scale
    font_scale = INFO_SCALE if font_scale == None else font_scale

    # Parse Information Height and BaseLine
    ( _, info_h ), info_base = cv2.getTextSize(CV_WIN, FONT_FACE, font_scale, FONT_THICK)

    # Get the Top Y position
    info_len = len(informations)
    info_padding = info_h * 0.8
    start_y = int((height - info_base) - (( info_h + info_padding ) * (info_len-1)))

    # Loop Each Information and Draw
    for idx, (key, val) in enumerate( informations.items() ):
        idx = idx + 1
        trg_x = 20
        trg_y = start_y + int((info_h + info_padding)*(info_len - idx))
        
        frame = put_text_border(    
            frame   = frame, 
            text    = "{:<6}: {}".format(key, val), 
            pos     = (trg_x, trg_y), 
            scale   = font_scale
        )
    
    return frame

def set_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="path to config", default="config.json")
    parser.add_argument("-a", "--app", help="enable app", action="store_true")
    parser.add_argument("--app-config", help="path to app config", default="application.json")
    args = parser.parse_args()
    return args




