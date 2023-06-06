#!/usr/bin/env python
# coding: utf-8

# In[ ]:
### Create Images to Numpy Arrays.
# Import the necessary libraries
import numpy as np
from PIL import Image
from numpy import asarray
import os
from os import listdir
import matplotlib.pyplot as plt
import tensorflow as tf 
from Model import *

model_save = False

from keras.preprocessing.image import ImageDataGenerator
aug_dict = dict(rotation_range=0.2,
                width_shift_range=0.05,
                height_shift_range=0.05,
                shear_range=0.05,
                zoom_range=0.05,
                horizontal_flip=True,
                fill_mode='nearest')

# define the right folders and 
batch_size=2 
train_path='./sample_data/train/'
#valid_path = './sample_data/valid/'
image_folder='images'
label_folder='labels'
image_color_mode='rgba'
label_color_mode='grayscale'
target_size=(256,256)
seed=1 # to make sure that the augmented images correspond to the right augmented labels. 


# define the data generators
image_datagen = ImageDataGenerator(**aug_dict)
label_datagen = ImageDataGenerator(**aug_dict)


image_generator = image_datagen.flow_from_directory(
    train_path,
    classes = [image_folder],
    class_mode = None,
    color_mode = image_color_mode,
    target_size = target_size,
    batch_size = batch_size,
    seed = seed)
label_generator = label_datagen.flow_from_directory(
    train_path,
    classes = [label_folder],
    class_mode = None,
    color_mode = label_color_mode,
    target_size = target_size,
    batch_size = batch_size,
    seed = seed)

train_generator = zip(image_generator, label_generator)
#valid_generator = zip(image_valid, mask_valid)


# With Generator function we can keep calling augmented images.
# This will be used when training the model.
# Transform the pixels in the label on the border of the roofs in the label
# to either 0 or 1. And if needed (just to be sure) resize the images.
def Generator(train_generator):
  for (img,mask) in train_generator:
      if(np.max(img) > 1): # just being sure to be scaled.
        img = img / 255.
        mask = mask /255.
      mask[mask > 0.5] = 1
      mask[mask <= 0.5] = 0
      yield (img,mask)
      
myGene = Generator(train_generator)

### Model Compile and Fit

input_img = Input((256,256, 4), name='img')
model = get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)
model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

epochs=10
steps_per_epoch=300  
                    
history = model.fit(myGene,steps_per_epoch=steps_per_epoch,epochs=epochs)
# In[ ]:



### SAVE MODEL

if model_save==True:
    
    path_to_save = './sample_data/saved_model_h5/'
#model.save(path_to_save)

    model.save(path_to_save+"my_h5_model.h5")
    
    print("Model is saved")



# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




