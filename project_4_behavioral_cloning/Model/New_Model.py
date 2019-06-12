# cd /home/workspace/CarND-Behavioral-Cloning-P3/Model/ && python New_Model.py
# python drive.py model.h5 run1
# python video.py run1


import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import PIL
import csv
from sklearn.model_selection import *
from sklearn.utils import shuffle
from keras.layers import *
from keras.models import *
from keras.layers.normalization import *
import re

# When using data collected from users computer
def parse_filename(name):
    pattern = re.compile('IMG(.*?).jpg')
    items = re.findall(pattern, name)
    return items


IMG_path = '/home/workspace/CarND-Behavioral-Cloning-P3/Data/data/'
samples=[]
"""
with open('/home/workspace/CarND-Behavioral-Cloning-P3/Data/data_1/driving_log.csv', 'r') as f:
    reader=csv.reader(f)
    for line in reader:
        samples.append(line)
    for i in range(0,len(samples)):
        for ii in range(0,3):
            string = samples[i][ii]
            file_name = parse_filename(string)
            file_name[0] = file_name[0].replace('\\','//')
            file_name[0] = IMG_path + file_name[0] + '.jpg'
            samples[i][ii] = file_name[0]
"""
# IMG/center_2016_12_01_13_30_48_287.jpg
with open('/home/workspace/CarND-Behavioral-Cloning-P3/Data/data/driving_log.csv', 'r') as f:
    reader = csv.reader(f)
    for line in reader:
        samples.append(line)
    for i in range(0, len(samples)):
        for ii in range(0, 3):
            string = samples[i][ii]
            file_path = IMG_path + string
            file_path = file_path.replace(" ", "")
            samples[i][ii] = file_path

train_samples, validation_samples = train_test_split(samples, test_size=0.2)


def generator(samples, batch_size=32):
    n_len=len(samples)
    while True:
        shuffle(samples)
        for i in range(0, n_len, batch_size):
            X=[]
            y=[]
            batches=samples[i:i+batch_size]
            for row in batches:
                correction = 0.2
                steering_center=float(row[3])
                steering_left=steering_center+correction
                steering_right=steering_center-correction
                
                img_center=np.asarray(PIL.Image.open(row[0]),dtype=np.uint8)
                img_left=np.asarray(PIL.Image.open(row[1]),dtype=np.uint8)
                img_right=np.asarray(PIL.Image.open(row[2]),dtype=np.uint8)
                img_flip=np.fliplr(img_center)# data augmentation with flip the image
                
                X.extend([img_center, img_left, img_right,img_flip])
                y.extend([steering_center, steering_left, steering_right,-steering_center])
               
            X=np.array(X)
            y=np.array(y)
            yield shuffle(X, y)

ch, width, height = 3, 320, 160

model=Sequential()

model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(height, width, ch)))
model.add(Cropping2D(cropping=((70,25),(0,0))))           

model.add(Convolution2D(24,5,5,subsample=(2,2)))
model.add(Activation('elu'))

model.add(Convolution2D(36,5,5,subsample=(2,2)))
model.add(Activation('elu'))

model.add(Convolution2D(48,5,5,subsample=(2,2)))
model.add(Activation('elu'))

model.add(Convolution2D(64,3,3))
model.add(Activation('elu'))

model.add(Convolution2D(64,3,3))
model.add(Activation('elu'))


model.add(Flatten())

model.add(Dense(100))
model.add(Activation('elu'))
model.add(Dropout(0.25))

model.add(Dense(50))
model.add(Activation('elu'))

model.add(Dense(10))
model.add(Activation('elu'))

model.add(Dense(1))


train_generator=generator(train_samples, 32)
valid_generator=generator(validation_samples, 32)
# len(train_samples)
model.compile(loss='mse',optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch = 1000, validation_data=valid_generator,   nb_val_samples=len(validation_samples), nb_epoch=3, verbose=1)
model.save('model.h5')
