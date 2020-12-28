# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 08:43:51 2019

@author: Yasin
"""
import os
from PIL import Image
import numpy as np
import pickle
import cv2

face_cascade=cv2.CascadeClassifier('cascades/haarcascade_frontalface_alt2.xml')


recognizer = cv2.face.LBPHFaceRecognizer_create();

curid=0
label_ids={}
y_labels=[]
x_train=[]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
img_dir=os.path.join(BASE_DIR,"img")

for root,dirs,files in os.walk(img_dir):
        for file in files:
            if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg"):
                path=os.path.join(root,file)
                label=os.path.basename(os.path.dirname(path)).replace(" ","-").lower()
               #print(label,path)
                if not label in label_ids:
                    label_ids[label]=curid
                    curid+=1
                id_=label_ids[label]
                pil_image=Image.open(path).convert("L")
                img_arr=np.array(pil_image,"uint8")
                #print(img_arr)
                faces=face_cascade.detectMultiScale(img_arr,scaleFactor=1.5,minNeighbors=5)
        
                
                for(x,y,w,h) in faces:
                    roi =img_arr[y:y+h,x:x+w]
                    x_train.append(roi)
                    y_labels.append(id_)
print(label_ids)           
with open("labels.pickle",'wb') as f:
    pickle.dump(label_ids,f)
                    
recognizer.train(x_train,np.array(y_labels))
recognizer.save("trainner.yml")