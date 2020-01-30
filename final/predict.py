from tensorflow.keras.models import load_model
from skimage import transform
from skimage import exposure
from skimage import io

import numpy as np
import argparse
import imutils
import random
import cv2
import os

# Load the traffic sign recognizer model
print("[INFO] loading model...")
model = load_model(os.path.join(os.getcwd(), 'final','MyModel.h5'))

# Grab the paths to the input images, shuffle them, and grab a sample
print("[INFO] predicting...")
imagepath = "Train/39/00039_00000_00009.png"

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

labelNames = open("map.csv").read().strip().split("\n")[1:]
labelNames = [l.split(",")[0] for l in labelNames]
j = preds.argmax(axis=1)[0]
label = labelNames[j]
    
image = cv2.imread(imagepath)
image = imutils.resize(image, width=128)
cv2.putText(image, label, (5,15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 2)
    
   
cv2.imwrite("/Speeeeed.png", image)
#cv2.imshow("m",image)
cv2.waitKey(0)
cv2.destroyAllWindows()
    
