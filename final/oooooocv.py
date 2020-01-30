import cv2
import numpy as np


def kaam(img):
    #sign_cascade = cv2.CascadeClassifier("cascade.xml")
    sign_cascade2=cv2.CascadeClassifier("Stopsign_HAAR_ 13Stages.xml")
	img = cv2.imread(img)
	#gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	#print(gray)
    signs=sign_cascade.detectMultiScale(img,scaleFactor=1.01,minNeighbors=5)
    print(type(signs))
    print(signs)

	count=0
	#cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
	#roi_gray=gray[y:y+h,x:x+w]
	#roi_color=img[y:y+h,x:x+w]
    for (x,y,w,h) in signs:
	    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
	    crop_img = img[y:y+h, x:x+w]
	if crop_img is None:
		crop_img=img
	#print(type(signs))
    #i=tuple(map(tuple,signs))
	#ii=totuple(signs)

	#print(x)

    #cv2.imshow('img',img)

#im = img.crop(i)
#cv2.imshow("jjj",im)


    k= cv2.waitKey(10000) & 0xFF 
    return crop_img
'''
while True:
	ret,img=cap.imread()
	#out.write(frame)
	gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	faces=face_cascade.detectMultiScale(gray,1.3,5)
	for (x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		roi_gray=gray[y:y+h,x:x+w]
		roi_color=img[y:y+h,x:x+w]
		eyes=eye_cascade.detectMultiScale(roi_gray)
		for (ex,ey,ew,eh) in eyes:
			cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)




	cv2.imshow('img',img)

k= cv2.waitKey(5) & 0xFF 
	if k==27:
		break
cap.release()
cv2.destroyAllWindows()
#fourcc=cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('output.avi',fourcc,20.0,(640,480))
'''