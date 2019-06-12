import numpy as np
import cv2
import math
import re
import matplotlib.pyplot as plt
import PIL
import csv
from sklearn.utils import shuffle
from sklearn.model_selection import *
from keras.layers import *
from keras.models import *
from keras.layers.normalization import *

# cd /home/workspace/CarND-Behavioral-Cloning-P3/Model/ && python Model.py

# Using local training data
def parse_filename(name):
    pattern = re.compile('D.*?IMG(.*?).jpg')
    items = re.findall(pattern, name)
    return items

ch, width, height = 3, 320, 160
IMG_path = '/home/workspace/CarND-Behavioral-Cloning-P3/Data/data_1/IMG/'
samples = []

# Import data, and chanllenge image stroge information
with open('/home/workspace/CarND-Behavioral-Cloning-P3/Data/data_1/driving_log.csv', 'r') as csvfiles:
    reader = csv.reader(csvfiles)
    samples.append([row for row in reader])
    for i in range(0,len(samples[0])):
        for ii in range(0,3):
            string = samples[0][i][ii]
            file_name = parse_filename(string)
            file_name[0] = file_name[0].replace('\\','//')
            file_name[0] = IMG_path + file_name[0] + '.jpg'
            samples[0][i][ii] = file_name[0]

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                correction = 0.2
                steering_center = float(row[3])
                steering_left = steering_center + correction
                steering_right = steering_center - correction
                
                image_center = np.asarray(PIL.Image.open(row[0]), dtype = np.unit8)
                image_left = np.asarray(PIL.Image.open(row[1]), dtype = np.unit8)
                image_right = np.asarray(PIL.Image.open(row[2]), dtype = np.unit8)
                image_flip = np.filter(image_center)
                
                images.extend([image_center, image_left, image_right, image_flip])
                angles.extend([steering_center, steering_left, steering_right, -steering_center])

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

# Set our batch size
batch_size=32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)


# Model constructure
model = Sequential()

# 1st Layer
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape = (height, width, ch)))
model.add(Cropping2D(cropping = ((70,25),(0,0))))

#2nd Layer
model.add(Convolution2D(24, 5, 5, subsample=(2, 2)))
#model.add(Activation('relu'))
model.add(Activation('elu'))
#model.add(BatchNormalization())
#model.add(MaxPooling2D(pool_size = (2,2), strides = 2, padding = 'same'))
#model.add(Dropout(0.5))

# 3rd Layer
model.add(Convolution2D(36, 5, 5, subsample=(2, 2)))
#model.add(Activation('relu'))
model.add(Activation('elu'))
#model.add(BatchNormalization())
#model.add(MaxPooling2D(pool_size = (2,2), strides = 2, padding = 'same'))
#model.add(Dropout(0.5))

# 4th Layer
model.add(Convolution2D(48, 5, 5, subsample=(2, 2)))
#model.add(Activation('relu'))
model.add(Activation('elu'))
#model.add(BatchNormalization())
#model.add(MaxPooling2D(pool_size = (2,2), strides = 2, padding = 'same'))
#model.add(Dropout(0.5))

# 5th Layer
model.add(Convolution2D(64, 3, 3))
#model.add(Activation('relu'))
model.add(Activation('elu'))
#model.add(BatchNormalization())
#model.add(MaxPooling2D(pool_size = (2,2), strides = 2, padding = 'same'))
#model.add(Dropout(0.5))

# 6th Layer
model.add(Convolution2D(64, 3, 3))
#model.add(Activation('relu'))
model.add(Activation('elu'))
#model.add(BatchNormalization())
#model.add(MaxPooling2D(pool_size = (2,2), strides = 2, padding = 'same'))
#model.add(Dropout(0.5))

# 7th Layer
model.add(Flatten())

# 8th Layer
model.add(Dense(100))
model.add(Activation('elu'))
model.add(Dropout(0.25))

# 9th Layer
model.add(Dense(50))
model.add(Activation('elu'))
#model.add(Dropout(0.25))

# 10th Layer
model.add(Dense(10))
model.add(Activation('elu'))
#model.add(Dropout(0.25))

# 11th Layer
model.add(Dense(1))

# Compile model
model.compile(loss='mse', optimizer='adam')

# model.fit_generator(train_generator, steps_per_epoch = len(train_samples), validation_data = validation_generator, validation_steps = len(validation_samples), epochs=5, verbose=1)
model.fit_generator(train_generator, samples_per_epoch = len(train_samples), validation_data = validation_generator,   nb_val_samples=len(validation_samples), nb_epoch = 5, verbose = 1)
model.save('model.h5')



