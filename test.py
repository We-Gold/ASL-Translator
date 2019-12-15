import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import numpy as np

model = load_model("model1.h5")

def load_image(img_path):
    img = image.load_img(img_path, target_size=(200, 200))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)         
    img_tensor /= 255.                                      
    return img_tensor

val_dir = os.listdir('data/validation')

def aslToChar(filename):
    alphabet = [ 
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K",
        "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W",
        "X", "Y", "Z", "del", "nothing", " "]
    return alphabet[np.argmax(model.predict(load_image(filename)))]

#for i in range(len(val_dir)):
#    filename = 'data/validation/'+val_dir[i]+'/'+val_dir[i]+'_test.jpg'
#    print(filename + ": ")
#    print(alphabet[np.argmax(model.predict(load_image(filename)))])
