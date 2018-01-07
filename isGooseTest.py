from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
import os
import glob
import shutil

image_name = raw_input("What is the image name? ")

# load the model we saved
model = load_model('isGoose.h5')


def readImg(filename):
    img              = load_img(filename, target_size=(64, 64))  
    imgArray         = img_to_array(img)  
    imgArrayReshaped = np.expand_dims(imgArray, axis=0)
    imgProcessed     = preprocess_input(imgArrayReshaped, mode='tf')
    return img, imgProcessed

img, arr = readImg(image_name)
probability = model.predict(arr)

if (probability[0][0] == 1):
	print "Not a goose!"
else:
	print "isGoose!"
