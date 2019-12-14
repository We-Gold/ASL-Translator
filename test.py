import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import numpy as np

model = load_model("model.h5")

def load_image(img_path):
    img = image.load_img(img_path, target_size=(200, 200))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)         
    img_tensor /= 255.                                      
    return img_tensor

val_dir = os.listdir('data/validation')

for i in range(len(val_dir)):
    print(np.argmax(model.predict(load_image('data/validation/'+val_dir[i]+'/'+val_dir[i]+'_test.jpg'))))
