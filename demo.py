# -*- coding: utf-8 -*-
import cv2, sys, os, time, logging
from pipeline import Source
from utils import *

# Define Source
SRC             = '/dev/video0'
SRC_TYPE        = 'Video'
LOGO            = 'assest/innodisk-logo.png'

if __name__ == '__main__':

    # Main
    args = set_argparse()

    # Defince logger and cv windows
    config_logging()
    
    # Load Configuration
    conf = read_json(args.config)

    # Update classes variable
    classes = conf["secondary"]["category"]
    thres = conf["primary"]["thres"]

    # Load All Network
    nets = load_multi_network(conf)

    # Instance Source Object and Application Object
    src = Source(SRC, SRC_TYPE)
    app = DiscountInfo(args.app_config) if args.app else None
    
    # Define Color
    trg_color = get_random_color()

    # First frame
    frame_idx = 0
    first_frame = src.get_first_frame()
    nets[ORG][HEIGHT], nets[ORG][WIDTH], nets[ORG][CHANNEL] = first_frame.shape
    
    # Rescale Logo
    logo = get_logo(
        path        = LOGO, 
        img_size    = (nets[ORG][HEIGHT], nets[ORG][WIDTH]), 
        scale       = LOGO_SCALE 
    )

    # Define CV Windows
    define_cv_window(
        title       = CV_WIN,
        first_frame = first_frame,
        fullscreen  = True
    )

    while True:

        # Get Input Data
        ret, image = src.get_frame()
        org_image = image.copy()
        t_start = time.time()

        # Pre-process
        temp_image = preprocess(image, (nets[PRI][WIDTH], nets[PRI][HEIGHT]) )

        # Inference : result[<OUT_BLOB_NAME>] -> 1, 1, 200, 7
        nets[PRI][RESULT] = nets[PRI][EXEC].infer( inputs = { nets[PRI][IN_BLOB][0]: temp_image } )

        # Get output layer name
        out_blob = nets[PRI][OUT_BLOB][0]

        # Run each bounding box
        for i in range(nets[PRI][RESULT][out_blob].shape[2]):
            
            # Parse bounding box
            (image_idx, label_idx, conf), bbox = parse_face_result( nets[PRI][RESULT][out_blob][0][0][i] )

            if conf > thres:

                # Rescaler bounding box
                x1, y1, x2, y2 = bbox_scaler(bbox, nets[ORG][HEIGHT], nets[ORG][WIDTH])

                # IOU: Get Face Area
                face_image = org_image[y1:y2, x1:x2]

                # Inference
                temp_image = preprocess(face_image, (nets[SEC][WIDTH], nets[SEC][HEIGHT]))
                nets[SEC][RESULT] = nets[SEC][EXEC].infer( inputs = { nets[SEC][IN_BLOB][0]: temp_image } )
        
                # Parse Result
                dets_text = ""
                gender, age = parse_gender_age(nets[SEC][RESULT], classes)
                        
                # Application
                if args.app:
                    discount_info = app(age, gender)
                    trg_color = app.get_info_color(discount_info)
                    dets_text += "{} ".format(discount_info)
                else:
                    dets_text = "Age {}, Gender {}".format(age, gender)       

                # Draw Information Text
                dets_text_ls = [ " - " + cnt.strip() for cnt in dets_text.split(",") ]

                # Draw Face Bounding Box
                org_image = cv2.rectangle(org_image, (x1, y1), (x2, y2), trg_color, 2)

                # Check if the Top out of limit
                org_image = draw_gender_age(
                    frame       = org_image, 
                    xyxy        = (x1, y1, x2, y2),
                    detections  = dets_text_ls,
                    org_size    = ( nets[ORG][HEIGHT], nets[ORG][WIDTH] ),
                    trg_color   = trg_color
                )

        # Resize Whole Image
        org_image = cv2.resize(org_image, (nets[ORG][WIDTH], nets[ORG][HEIGHT]))

        if not args.no_logo:
            org_image = add_logo(org_image, logo)

        # Get Temperature
        temp = get_cpu_temp()
        
        # Draw FPS
        t_end = time.time()
        fps = int((1/(t_end - t_start)))
        informations = { 
            "FPS"   : fps,
            "TEMP"  : temp
        } 

        org_image = draw_information(
            frame = org_image, 
            informations = informations,
            font_scale = 1.6
        )
        
        # Show result
        cv2.imshow(CV_WIN, org_image)  
        
        # Define Button Event
        key = cv2.waitKey(1)

        if key in [ ord('q'), 27 ]: 
            break

        elif key == ord('c'):
            if args.app:
                app.generate_palette()
            else:
                trg_color = get_random_color()

        elif key in [ ord('a'), 1, 123 ]: 
            FULL_SCREEN = not FULL_SCREEN
            cv2.setWindowProperty(CV_WIN,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN if FULL_SCREEN else cv2.WINDOW_NORMAL )

        # Update Frame Index
        frame_idx += 1    
        
    cv2.destroyAllWindows()
    logging.warning("Quit")