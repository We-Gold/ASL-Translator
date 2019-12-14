from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

import os
import numpy as np
import matplotlib.pyplot as plt

PATH = "data"

train_dir = PATH + "/train"
validation_dir = PATH + "/validation"

batch_size = 128
epochs = 1
IMG_HEIGHT = 200
IMG_WIDTH = 200

total_train = 0
total_val = len(os.listdir(validation_dir))

for token in os.listdir(train_dir):
    total_train += len(os.listdir(train_dir + "/" + token))

train_image_generator = ImageDataGenerator(rescale=1./255)
validation_image_generator = ImageDataGenerator(rescale=1./255)

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,directory=train_dir,shuffle=True,target_size=(IMG_HEIGHT, IMG_WIDTH),class_mode='categorical')
val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,directory=validation_dir,shuffle=True,target_size=(IMG_HEIGHT, IMG_WIDTH),class_mode='categorical')

model = Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Dropout(0.5),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(),
    Dropout(0.5),
    Conv2D(16, (3,3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(29, activation='softmax')
])

filepath="checkpoints/weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy']) #may switch to sparse

model.summary()

history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=50,#total_train // batch_size,
    epochs=epochs
)

model.save("model.h5")
