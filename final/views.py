from django.shortcuts import render
#from oooooocv import kaam
# Create your views here.

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from skimage import transform
from skimage import exposure
from skimage import io
import argparse
import imutils
import random
from django.contrib import messages
import os
'''
def kaam(img):
sign_cascade = cv2.CascadeClassifier("final/cascade.xml")
    sign_cascade2=cv2.CascadeClassifier("final/Speedlimit_HAAR_ 13Stages.xml")
    #sign_cascade3=cv2.CascadeClassifier("Stopsign_HAAR_19Stages.xml")
#eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

#print(face_cascade)

    img = cv2.imread(img)
#gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#print(gray)
    signs=sign_cascade2.detectMultiScale(img,scaleFactor=1.01,minNeighbors=5)
    signs=list(signs)
    print(type(signs))
    print(signs)
#cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
#roi_gray=gray[y:y+h,x:x+w]
#roi_color=img[y:y+h,x:x+w]
    if not signs:
	    pass
    else:
        for (x,y,w,h) in signs:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            crop_img = img[y:y+h, x:x+w]
            
    #print(type(crop_img))   
    
#print(type(signs))
    i=tuple(map(tuple,signs))
#ii=totuple(signs)

	#print(x)
    

#im = img.crop(i)
#cv2.imshow("jjj",im)'''
def kaam(request):
    img = request.POST.get("myfile")
    # Load the traffic sign recognizer model
    print("[INFO] loading model...")
    model = load_model("D:/image_processing/sih2020/final/MyModel.h5")
    #export_path = os.path.join(os.getcwd(), 'MyModel.h5')
    # Grab the paths to the input images, shuffle them, and grab a sample
    print("[INFO] predicting...")
    '''sign_cascade2=cv2.CascadeClassifier("Stopsign_HAAR_19Stages.xml")
    img = cv2.imread(img1)
	#gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	#print(gray)
    signs=sign_cascade2.detectMultiScale(img,scaleFactor=1.01,minNeighbors=5)
    print(type(signs))
    print(signs)

	#cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
	#roi_gray=gray[y:y+h,x:x+w]
	#roi_color=img[y:y+h,x:x+w]
    for (x,y,w,h) in signs:
	    #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
	    crop_img = img[y:y+h, x:x+w]
    if crop_img is None:
        crop_img = img
    '''
    imagepath = img

    # load the image using scikit learn
    image = io.imread(imagepath)
    # Resize
    image = transform.resize(image,(32,32))
    # CLAHE
    image = exposure.equalize_adapthist(image, clip_limit = 0.1)
    # Scale [0,1]
    image = image.astype("float32") /255.0
    # Add a dimension to the image
    image = np.expand_dims(image, axis=0)
        
    # Make a prediction and grab the class with highest probability
    preds = model.predict(image)
    #print(preds)

    labelNames = open("D:\image_processing\sih2020\Map.csv").read().strip().split("\n")[1:]
    labelNames = [l.split(",")[0] for l in labelNames]
    j = preds.argmax(axis=1)[0]
    label = labelNames[j]
    image = cv2.imread(imagepath)
    image = imutils.resize(image,width=128)
    print('#########################################')
    print(label)
    print('###################################')
    sti = label
    cv2.putText(image, label, (5,15), cv2.FONT_HERSHEY_TRIPLEX, 0.45, (255,0,100), 2)
    messages.add_message(request, messages.INFO,sti)
    cv2.imwrite("final\static\images\Speeeeed.png", image)
    cv2.destroyAllWindows()
    return render(request,'home.html',{'name':sti})


def final2020(request):
    #print(fil.value)
    #print("myfile")

    #if 'myfile' in request.POST:
    '''if request.method == 'POST' and request.FILES["myfile"]:
        post = request.method == 'POST'
        myfile = request.FILES['myfile']
        img = image.load_img(myfile)
        #value = 'value' in request.POST and request.POST['myfile']
        #value=str(myfile)
        kaam(img)'''
    
    return render(request,'home.html')
    '''
from django.shortcuts import render
#from oooooocv import kaam
# Create your views here.

import cv2
import numpy as np


def kaam(img):
    sign_cascade = cv2.CascadeClassifier("final/cascade.xml")
    sign_cascade2=cv2.CascadeClassifier("final/Speedlimit_HAAR_ 13Stages.xml")
#eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

#print(face_cascade)

    img = cv2.imread(img)
#gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#print(gray)
    signs=sign_cascade2.detectMultiScale(img,scaleFactor=1.01,minNeighbors=5)
    print(type(signs))
    print(signs)
#cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
#roi_gray=gray[y:y+h,x:x+w]
#roi_color=img[y:y+h,x:x+w]
    for (x,y,w,h) in signs:
	    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
	    crop_img = img[y:y+h, x:x+w]
#print(type(signs))
    i=tuple(map(tuple,signs))
#ii=totuple(signs)

	#print(x)

    cv2.imshow('img',crop_img)

#im = img.crop(i)
#cv2.imshow("jjj",im)

    k= cv2.waitKey(1000) & 0xFF 

def final2020(request):
    #print(fil.value)
    value=request.GET['myfile']
    print(value)
    values=str(value)
    
    kaam(values)
    
    return render(request,'home.html')
'''