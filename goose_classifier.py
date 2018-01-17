from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import EarlyStopping

import os
import glob
import shutil

import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input

def readImg(filename):
    img              = load_img(filename, target_size=(64, 64))  
    imgArray         = img_to_array(img)  
    imgArrayReshaped = np.expand_dims(imgArray, axis=0)
    imgProcessed     = preprocess_input(imgArrayReshaped, mode='tf')
    return img, imgProcessed

earlystop = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=5,
                          verbose=1, mode='auto')

callbacks_list = [earlystop]

gooseIdentifier = Sequential()

gooseIdentifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

gooseIdentifier.add(MaxPooling2D(pool_size = (2, 2)))

gooseIdentifier.add(Conv2D(64, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

gooseIdentifier.add(Conv2D(64, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

gooseIdentifier.add(MaxPooling2D(pool_size = (2, 2)))

gooseIdentifier.add(Flatten())

gooseIdentifier.add(Dense(units=512, activation='relu'))

gooseIdentifier.add(Dropout(rate=0.5))

gooseIdentifier.add(Dense(units=1, activation='sigmoid'))

gooseIdentifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
				shear_range = 0.2, 
				zoom_range = 0.2, 
				horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('training_set', 
												target_size = (64, 64),
												batch_size = 32,
												class_mode = 'binary')

test_set = test_datagen.flow_from_directory('test_set',
											target_size = (64, 64),
											batch_size = 32,
											class_mode = 'binary')

gooseIdentifier.fit_generator(training_set, 
								steps_per_epoch = 58,
								epochs = 50, 
								validation_data = test_set,
								validation_steps = 19,
								callbacks = callbacks_list)

for filename in glob.glob('validation_set/Goose/*'):
    img, arr = readImg(filename)
    probability = gooseIdentifier.predict(arr)
    if probability[0][0] == 1:
    	shutil.move(filename, "Validation_False Negatives/" + os.path.basename(filename))
    else:
    	shutil.move(filename, "Validation_True Positives/" + os.path.basename(filename))

for filename in glob.glob('validation_set/NotGoose/*'):
    img, arr = readImg(filename)
    probability = gooseIdentifier.predict(arr)
    if probability[0][0] == 1:
    	shutil.move(filename, "Validation_True Negatives/" + os.path.basename(filename))
    else:
    	shutil.move(filename, "Validation_False Positives/" + os.path.basename(filename))

gooseIdentifier.save("isGoose.h5")

