# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 00:21:36 2020

@author: avnis
"""

#Keras
from tensorflow import keras

# Import of keras model and hidden layers for CNN
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten, Dropout

#Image handling libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd

#Sklearn libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# data visualisation and manipulation
import matplotlib.pyplot as plt
from matplotlib import style

#Initialize a list of paths for images
imagepaths = []

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        path = os.path.join(dirname, filename)
        imagepaths.append(path)
        
IMG_SIZE=128
X=[]
y=[]
for image in imagepaths:
    try:
        img = cv2.imread(image,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))

        X.append(np.array(img))
        if(image.startswith('/kaggle/input/pothole-detection-dataset/normal/')):
            y.append('NORMAL')
        else:
            y.append('POTHOLES')
    except:
        pass

from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


le=LabelEncoder()
Y=le.fit_transform(y)
Y=to_categorical(Y,2)
X=np.array(X)


x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=5)

# Create a CNN Sequential Model
model = Sequential()

model.add(Conv2D(32, (5,5), activation = 'relu', input_shape=(128,128,3)))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64, (3, 3), activation='relu')) 
model.add(MaxPooling2D((2, 2)))

# add new
# model.add(Conv2D(64, (3, 3), activation='relu')) 
# model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu')) 
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu')) 
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu')) 
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
# new darta
model.add(Dropout(0.2))

model.add(Dense(2, activation='softmax'))

from sklearn.externals import joblib 
# # Save the model as a pickle in a file 
# joblib.dump(model, 'outputfile.pkl') 
  
# Load the model from the file 
knn_from_joblib = joblib.load('/kaggle/input/outputfile/outputfile.pkl')  
  



img = cv2.imread('/kaggle/input/testset/images.jpg',cv2.IMREAD_COLOR)
img = cv2.resize(img, (128,128))

a=np.array(img)
a.resize(1,128,128,3)
# a.reshape(a.shape[0], 1, 128, 128)
# Use the loaded model to make predictions 
predicted_value=knn_from_joblib.predict(a)

for value in predicted_value:
    if(value[0]>value[1]):
        print("Normal")
    else:
        print("PothHole")        


