from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
import torch
import time
import cv2
import pytesseract
import easyocr
import time
import re
import numpy as np
# UPDATES
import sys
from difflib import SequenceMatcher
from imutils.video import FileVideoStream
import imutils
import argparse
import csv
import os
import uuid
#new
from .Gmail import Create_Service
import base64
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import subprocess
from django.conf import settings


#new mail content
CLIENT_SECRET_FILE = r'C:\Users\S.SANTHOSH\OneDrive\Desktop\proj\tinycoders-20220824T180516Z-001\tinycoders\detector\client_secret.json'
API_NAME = 'gmail'
API_VERSION = 'v1'
SCOPES = ['https://mail.google.com/']

service = Create_Service(CLIENT_SECRET_FILE, API_NAME, API_VERSION, SCOPES)

# Create your views here.

def index(request):
    return render(request, 'index.html')
def output(request):
    if request.method == 'GET':
        subprocess.Popen(r'explorer /select,"C:\files\temp"')
        return render(request, 'index.html')


def startwebcam(request):
    return HttpResponse("Hello, world. You're at the polls index.")


def imagedetect(request):
    if request.method == 'POST':
        # file = request.POST['imagepath']
        uploaded_file = request.FILES['imagepath']
        print(uploaded_file)
        fs = FileSystemStorage()
        fs.save(uploaded_file.name, uploaded_file)

        main(img_path="c:\\files\\" + uploaded_file.name,
        im_out="c:\\files\\"+uploaded_file.name+"_result.jpg")
    return render(request, 'index.html')


def videodetect(request):
    if request.method == 'POST':
        uploaded_file = request.FILES['videopath']
        print(uploaded_file)
        fs = FileSystemStorage()
        fs.save(uploaded_file.name, uploaded_file)
        if os.path.exists("c:\\files\\result.mp4"):
            os.remove("c:\\files\\result.mp4")
        main(vid_path="c:\\files\\" + uploaded_file.name,
             vid_out="c:\\files\\"+uploaded_file.name+"_result.mp4")
        if os.path.exists("c:\\files\\"+uploaded_file.name+"_result.mp4"):
            print("success")
            filename = uploaded_file.name + "_result.mp4"
            file_path = os.path.join(settings.MEDIA_ROOT, filename)
            context = {'data': file_path}
            return render(request, 'index.html', context)
    return render(request, 'index.html')

def webcamdetect(request):
    if request.method == 'POST':
        main(vid_path=0, vid_out='{}.mp4'.format(uuid.uuid1()))

    return render(request, 'index.html')

def sleepy( seconds):
    time.sleep(seconds)

def DetectR(frame, model):
    frame = [frame]
    print(f" $ Detecting. . . ")
    results = model(frame)
    labels, cordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    return labels, cordinates


# OCR
EZOCR = easyocr.Reader(['en'])
OCR_TH = 0.2
l=["MPO7CH4404","GPI9 XDO","LF64 KOA","TN 63 DB 5481","GVII EXP","T333 JEL","NA54 KKP","GF67 KVT","AY03 DOY","RJIB VOD","H659 ODN","21 BH 2345 AA","LG69 FPE","WXI9 SHV","DL1C H8 337","HROI AD 3205","PB23GO 261","DL3C S 1166","JK18 0515","PB 44 B 0606","UP 27T 0188","HR 7K 6422","HR 26 5383"]
npl=[]
NP=[]






# PLOTS AND RESULTS
def plot_boxes(results, frame, classes):
    labels, cord = results
    dets = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]

    print(f" $ found {dets} detections. . . ")
    print(f" $ looping values... ")

    for i in range(dets):
        row = cord[i]
        if row[4] >= 0.55:  # threshold value
            print(f" $ Extracting BBox values... ")
            x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(
                row[3] * y_shape)  ## BBox coordniates
            textval = classes[int(labels[i])]
            # cv2.imwrite("./output/dp.jpg",frame[int(y1):int(y2), int(x1):int(x2)])

            coords = [x1, y1, x2, y2]

            plate_num = reco_plate(img=frame, coords=coords, reader=EZOCR, region_threshold=OCR_TH)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  ## BBox
            cv2.rectangle(frame, (x1, y1 - 20), (x2, y1), (0, 255, 0), -1)  ## for text label background
            cv2.putText(frame, f"{plate_num}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # cv2.imwrite("./output/np.jpg",frame[int(y1)-25:int(y2)+25, int(x1)-25:int(x2)+25])

    return frame


# OCR FUNCTION

def reco_plate(img, coords, reader, region_threshold):
    # global valz
    # global glplate
    # Getting Co-ordinates
    xmin, ymin, xmax, ymax = coords
    Lplate = img[int(ymin):int(ymax), int(xmin):int(xmax)]
    print(type(Lplate))
    ocr_out = reader.readtext(Lplate)
    # textT=pytesseract.image_to_string(Lplate)
    # print("tesseract:",textT)

    text = text_filter(region=Lplate, ocr_out=ocr_out, region_threshold=region_threshold)
    if len(text) == 1:
        text = text[0].upper()
    # valz=text
    # glplate=Lplate
    save_results(text, Lplate, 'C:\\Users\\S.SANTHOSH\\OneDrive\\Desktop\\proj\\tinycoders-20220824T180516Z-001\\tinycoders\\excels\\realtimeresults_4.csv', 'numpl_imgs')
    return text


# FILTERING PLATES
def text_filter(region, ocr_out, region_threshold):
    rectangle_size = region.shape[0] * region.shape[1]

    plates = []
    print(ocr_out)

    for result in ocr_out:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))

        if length * height / rectangle_size > region_threshold:
            plates.append(result[1])
    return plates


def save_results(text, Lplate, csv_filename, folder_path):
    img_name = '{}.jpg'.format(uuid.uuid1())
    global NP

    for i in l:
        s=SequenceMatcher(None,i,text)
        if s.ratio()>0.70:
            if i not in npl:
                npl.append(i)
                cv2.imwrite(os.path.join(folder_path, img_name), Lplate)
                with open(csv_filename, mode='a', newline='') as f:
                    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow([img_name,text,i])
                break
        else:
            NP.append(text)
            if len(NP)==100:
                print(NP)
                NP_string=' '.join(map(str,NP))
                emailMsg = 'Number plate not found in the database: '+NP_string
                mimeMessage = MIMEMultipart()
                mimeMessage['to'] = 'uit20202@rmd.ac.in'
                mimeMessage['subject'] = 'test'
                mimeMessage.attach(MIMEText(emailMsg, 'plain'))
                raw_string = base64.urlsafe_b64encode(mimeMessage.as_bytes()).decode()
                
                message = service.users().messages().send(userId='me', body={'raw': raw_string}).execute()
                print(message)
                


# MAIN FUNCTION
def main(img_path=None, vid_path=None,im_out=None, vid_out=None):
    print(img_path)
    print(f" $ Loading Custom model... ")
    model = torch.hub.load(r'C:\Users\S.SANTHOSH\OneDrive\Desktop\YOLO\yolov5', 'custom', source='local',
                           path="C:\\Users\\S.SANTHOSH\\OneDrive\\Desktop\\YOLO\\yolov5\\custom_models\\1.pt",
                           force_reload=True)

    classes = model.names
    
    
    # E:\opencv\venv\Lib\site - packages\yolov5\hubconf.py
    # FOR IMAGES
    if img_path != None:
        print(f" $ Processing image: {img_path}")
        out_img = "c://files//output//result.jpg"
        print(out_img)
        frame = cv2.imread(img_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = DetectR(frame, model=model)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        frame = plot_boxes(results, frame, classes=classes)
        # print (frame)
        # cv2.imwrite(f"{out_img}", frame)
        cv2.namedWindow("ANPR-IMG", cv2.WINDOW_NORMAL)

        while True:
            cv2.imshow("ANPR-IMG", frame)
            if cv2.waitKey(0):
                cv2.imwrite(im_out, frame)
                #cv2.destroyWindow("ANPR-IMG")
                break
        #
        #    if cv2.waitKey(5) & 0xFF == ord('q'):
        #         print(f" $ Finishing... ")
        #
        #     cv2.imwrite(f"{out_img}", frame)
        #
        #     break

    # FOR VIDEOS
    elif vid_path != None:
        print(f" $ Proceesing video: {vid_path}")
        

        cap = cv2.VideoCapture(vid_path)
        fvs = FileVideoStream(vid_path).start()
        # time.sleep(1.0)

        if vid_out:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            codec = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(vid_out, codec, fps, (w, h))

        frame_no = 1
        count = 0
        l1 = []

        cv2.namedWindow("ANPR", cv2.WINDOW_NORMAL)
        while fvs.more():
            frame = fvs.read()
            if frame is None:
                break

            print(f" $ Working with frame {frame_no} ")

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # frameR=cv2.resize(frame, (640,640), interpolation= cv2.INTER_LINEAR)
            # start_time = time.time()
            results = DetectR(frame, model=model)
            # print("results finished --- %s seconds ---" % (time.time() - start_time))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = plot_boxes(results, frame, classes=classes)
            cv2.imshow("ANPR", frame.astype(np.uint8))
            # save_results(valz, glplate, 'realtimeresults_6.csv', 'Detection_Images')

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

# MAIN FUNCTION CALL
# MAIN FUNCTION CALL

# custom video
# main(vid_path="C:\\Users\\nice\\Desktop\\YOLO\\yolov5\\test_inputs\\testx\\1440.mp4",
#      vid_out="C:\\Users\\nice\\Desktop\\YOLO\\yolov5\\tested_outputs\\vid_s_42.mp4")

# for web cam
#main(vid_path=0,vid_out="webcam_1t.mp4")

# for image
# main(img_path="test3.jpg")
