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
from keras.callbacks import ModelCheckpoint

# Download the mobile net pretrained base.
base_model = MobileNet(weights = 'imagenet',include_top = False) 

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024,activation = 'relu')(x) 
x = Dense(512,activation = 'relu')(x)
preds = Dense(29,activation = 'softmax')(x)

# Initialize the full model.
model = Model(inputs = base_model.input,outputs = preds)

for layer in model.layers[:20]:
    layer.trainable = False
for layer in model.layers[20:]:
    layer.trainable = True

# Setup the data generator, and preprocess the data.
train_datagen = ImageDataGenerator(rescale = 1./255, rotation_range = 15, width_shift_range = 0.25, height_shift_range = 0.25)

train_generator = train_datagen.flow_from_directory('data/train',
                                                 target_size = (200,200),
                                                 color_mode = 'rgb',
                                                 batch_size = 32,
                                                 class_mode = 'categorical',
                                                 shuffle = True)

model.compile(optimizer = 'Adam',loss = 'categorical_crossentropy',metrics = ['accuracy'])

checkpoints = ModelCheckpoint("checkpoints/weights.{epoch:02d}.h5",
                                          save_weights_only = False,
                                          verbose = 1)

step_size_train = train_generator.n//train_generator.batch_size

# Train and save the model
model.fit_generator(generator = train_generator,
                   steps_per_epoch = step_size_train,
                   epochs = 4,
                   callbacks = [checkpoints])

model.save("model2.h5")