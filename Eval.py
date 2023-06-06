#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from PIL import Image
from numpy import asarray
import os
from os import listdir
import matplotlib.pyplot as plt
import tensorflow as tf
import skimage.io as io
from Model import *

#Data Preparation
#from keras.models import *
#from keras.layers import *
#from keras.optimizers import *
#from keras import backend as keras
#from keras.preprocessing.image import ImageDataGenerator

folder_dir = "./sample_data/test/images/"
test_images=[]
real_test=[]
file_names = []
for images in os.listdir(folder_dir):
 
    # check if the image ends with png
    if (images.endswith(".png")):
        file_names.append(images)
        #normalized pixel outputs, between 0 and 1
        img = plt.imread(os.path.join(folder_dir+images))
        #real image
        real_img = Image.open(folder_dir+images)
        
        real_test.append(real_img)
        test_images.append(img)
        
test_=np.array(test_images)

path_to_save = './sample_data/saved_model_h5/'

#test_2=model.predict(test_,steps=len(file_names))

def saveResult(save_path,results,filenames):
    for i,item in enumerate(results):
        img = item[:,:,0]
        img_name = filenames[i]
        io.imsave(os.path.join(save_path, img_name),img)
        
# LOAD MODEL AND TEST 
model = tf.keras.models.load_model(path_to_save+"my_h5_model.h5")
test_2=model.predict(test_,steps=len(file_names))
saveResult("./sample_data/test/labels_2/",(test_2*255).astype(np.uint8), file_names)