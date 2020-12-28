# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 07:17:56 2019

@author: Yasin
"""
import pickle
import cv2
face_cascade=cv2.CascadeClassifier('cascades/haarcascade_frontalface_alt2.xml')
recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")
labels={"person_name":1}
with open("labels.pickle",'rb') as f:
    og_labels=pickle.load(f)
    labels={v:k for k,v in og_labels.items()}
cap = cv2.VideoCapture(0)
#i=0
while(True):
    ret,frame=cap.read()
    
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)
    for(x,y,w,h) in faces:
        #print(x,y,w,h)
        
        roi_gray=gray[y:y+h,x:x+w]
        roi_color=frame[y:y+h,x:x+w]
        id_,conf=recognizer.predict(roi_gray)
        if conf<+45: # and conf<=85:
            font=cv2.FONT_HERSHEY_SIMPLEX
            name=labels[id_]
            color=(255,255,255)
            stroke=2
            cv2.putText(frame,name,(x-10,y-10),font,1,color,stroke,cv2.LINE_AA)
        #img_item="myimg"+str(i)+".png"
        #cv2.imwrite(img_item,roi_gray)
        #i+=1
        color=(255,0,0)
        stroke=2
        width=x+w
        height=y+h
        cv2.rectangle(frame,(x,y),(width,height),color,stroke)
        
        
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF==ord('q'):
      break
cap.release()
cv2.destroyAllWindows()
