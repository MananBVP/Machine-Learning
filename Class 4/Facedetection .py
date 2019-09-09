# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 19:23:27 2019

@author: Manan
"""

import cv2
import numpy as np

camera = cv2.VideoCapture(0)
BASE_DIR = "./Data/"

name = input("Enter your name : ")
face_data = []

count=0

while True:
    ret,img = camera.read()
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    
    if ret==False:
        continue
    
    faces = face_detector.detectMultiScale(img,1.3,5)
    
    if(len(faces)==0):
        print("0 faces detected")
        continue

faces = sorted(faces,key = lambda X:X[2]*X[3],reverse = True)        

for face in faces:
    x,y,w,h = faces[0]
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    
    cropped_face = img[y:y+h,x:x+w]
    cropped_face = cv2.resize(cropped_face,(100,100))
    
    cv2.imshow("Title",img)
    cv2.imshow("Cropped Face",cropped_face)
        
    key = cv2.waitKey(1) & 0xFF
    if key==ord('q'):
        break
        
    count+=1 
    if count%10==0:
        face_data.append(cropped_face)
        print("Saving Pic",(count/10))
    
    
camera.release()
cv2.destroyAllWindows()

face_data = np.asarray(face_data)
np.save(BASE_DIR+name+".npy",face_data)