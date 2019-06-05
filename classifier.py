import pandas as pd
import numpy as np
import os
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam
import cv2
import glob
from random import shuffle

base_model=MobileNet(weights='imagenet',include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.

x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x=Dense(1024,activation='relu')(x) #dense layer 2
x=Dense(512,activation='relu')(x) #dense layer 3
preds=Dense(1,activation='sigmoid')(x) #final layer with softmax activation

model=Model(inputs=base_model.input,outputs=preds)
for i,layer in enumerate(model.layers):
  print(i,layer.name)

for layer in model.layers[:20]:
    layer.trainable=False
for layer in model.layers[20:]:
    layer.trainable=True

img_array = []
label_array = []
images = glob.glob('guards/*.jpg')
for i in images:
    img_array.append(preprocess_input(cv2.resize(cv2.imread(i),(224,224))))
    label_array.append(1)

unknown_images = glob.glob('joe_gatto/*.jpg')
for i in unknown_images:
    img_array.append(preprocess_input(cv2.resize(cv2.imread(i),(224,224))))
    label_array.append(0)

print('image dataset shape:',np.array(img_array).shape,'label dataset shape:',np.array(label_array).shape)
c = list(zip(img_array,label_array))
shuffle(c)
x_train,y_train = zip(*c)
model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(np.array(x_train),np.array(y_train),epochs=100,batch_size=32,verbose=1)
model.save('police_2.hdf5')
print('model saved as police_2.hdf5')
