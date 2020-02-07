# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 15:02:17 2020

@author: avnis
"""
import cv2
import numpy as np
import joblib

model= joblib.load('outputfile.pkl') 
img = cv2.imread('p.jpg',cv2.IMREAD_COLOR)
img = cv2.resize(img, (128,128))
a=np.array(img)

a.resize(1,128,128,3)
print(a)
# a.reshape(a.shape[0], 1, 128, 128)
# Use the loaded model to make predictions 
predicted_value=model.predict(a)
for value in predicted_value:
    
    if(value[0]>value[1]):
        print("Normal")
    else:
        print("PothHole")