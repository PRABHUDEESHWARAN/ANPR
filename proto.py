

import torch
import cv2
import easyocr
import pytesseract
import time
import re
import numpy as np
from imutils.video import FileVideoStream
import sys
from difflib import SequenceMatcher
import imutils
import argparse
import csv
import os
import uuid



def DetectR (frame, model):
    frame = [frame]
    print(f" $ Detecting. . . ")
    results = model(frame)
    labels, cordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    return labels, cordinates

 
EZOCR = easyocr.Reader(['en'])
OCR_TH = 0.2





def plot_boxes(results, frame,classes):

    
    labels, cord = results
    dets = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]

    print(f" $ found {dets} detections. . . ")
    print(f" $ looping values... ")


    for i in range(dets):
        row = cord[i]
        if row[4] >= 0.55: 
            print(f" $ Extracting BBox values... ")
            x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
            textval = classes[int(labels[i])]
            

            coords = [x1,y1,x2,y2]

            plate_num= reco_plate(img = frame, coords= coords, reader= EZOCR, region_threshold= OCR_TH)


            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255, 0), 2)
            cv2.rectangle(frame, (x1, y1-20), (x2, y1), (0,255,0), -1) 
            cv2.putText(frame, f"{plate_num}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255), 2)

            




    return frame




def reco_plate(img, coords,reader,region_threshold):
    xmin, ymin, xmax, ymax = coords
    Lplate = img[int(ymin):int(ymax), int(xmin):int(xmax)]
    
    print(type(Lplate)) 
    ocr_out = reader.readtext(Lplate)
    text = text_filter(region=Lplate, ocr_out=ocr_out, region_threshold= region_threshold)
    if len(text) ==1:
        text = text[0].upper()
    save_results(text, Lplate, 'uses\\realtimeresults_12.csv', 'numpl_imgs')
    return text

    



def text_filter(region, ocr_out, region_threshold):
    rectangle_size = region.shape[0]*region.shape[1]
    
    plates = [] 
    print(ocr_out)

    for result in ocr_out:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))
        
        if length*height / rectangle_size > region_threshold:
            plates.append(result[1])
    return plates

def save_results(text, Lplate, csv_filename, folder_path):
    img_name = '{}.jpg'.format(uuid.uuid1())
    #cv2.imwrite(os.path.join(folder_path, img_name), Lplate)
    with open(csv_filename, mode='a', newline='') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow([img_name,text])


    


def main(img_path=None, vid_path=None,im_out=None,vid_out =None):

    print(f" $ Loading Custom model... ")
    model =  torch.hub.load(r'C:\Users\S.SANTHOSH\OneDrive\Desktop\YOLO\yolov5', 'custom', source ='local', path="C:\\Users\\S.SANTHOSH\\OneDrive\\Desktop\\YOLO\\yolov5\\custom_models\\1.pt",force_reload=True)

    classes = model.names

    if sys.argv[1]=="None":
        img_path=None
    if len(sys.argv[2])==1:
        vid_path=int(sys.argv[2])
    if sys.argv[2]=="None":
        vid_path=None



    if img_path != None:
        print(f" $ Processing image: {img_path}")
        frame = cv2.imread(img_path)
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        
        results = DetectR(frame, model = model)   

        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

        frame = plot_boxes(results, frame,classes = classes)
        

        cv2.namedWindow("ANPR-IMG", cv2.WINDOW_NORMAL)

        while True:
            cv2.imshow("ANPR-IMG", frame)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                print(f" $ Finishing... ")

                cv2.imwrite(im_out,frame)
                cv2.destroyWindow("ANPR-IMG")

                break

    
    if vid_path !=None:
        print(f" $ Proceesing video: {vid_path}")

        cap = cv2.VideoCapture(vid_path)
        fvs = FileVideoStream(vid_path).start()
        
        if vid_out:

            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            codec = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(vid_out, codec, fps, (w, h))

        frame_no = 1
        count=0
        l1=[]

        cv2.namedWindow("ANPR", cv2.WINDOW_NORMAL)
        while fvs.more():
            frame = fvs.read()
            if frame is None:
                break
            
            print(f" $ Working with frame {frame_no} ")

            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

            results = DetectR(frame, model = model)
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            frame = plot_boxes(results, frame,classes = classes)
            cv2.imshow("ANPR",frame.astype(np.uint8))
                    
            if vid_out:
                print(f" $ Saving video... ")
                out.write(frame)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
            frame_no += 1
                
        
        print(f" $ Finishing... ")
        out.release()

        cv2.destroyAllWindows()
        fvs.stop()




main(img_path="C:\\Users\\S.SANTHOSH\\OneDrive\\Desktop\\YOLO\\yolov5\\test_inputs\\%s"%sys.argv[1],vid_path="C:\\Users\\S.SANTHOSH\\OneDrive\\Desktop\\YOLO\\yolov5\\test_inputs\\%s"%sys.argv[2],im_out="C:\\Users\\S.SANTHOSH\\OneDrive\\Desktop\\YOLO\\yolov5\\tested_outputs\\%s"%sys.argv[3],vid_out="C:\\Users\\S.SANTHOSH\\OneDrive\\Desktop\\YOLO\\yolov5\\tested_outputs\\%s"%sys.argv[4])


