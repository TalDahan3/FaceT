"""
import keras
import numpy as np
import os as os
import matplotlib.pyplot as pyplot
import datetime
from datetime import timedelta
from PIL import Image
import requests
from io import BytesIO
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
"""
################################

import json

#nn model
import keras
import FaceDetectionClass 
from FaceDetectionClass import FaceDetectionModel
from FaceDetectionClass import BaseModel
from DataGenerator import DataGenerator

#image preprocessing
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from PIL import Image

#visualization
import matplotlib.pyplot as plt
import matplotlib.patches as patches

#math
import numpy as np
###############################################
def DownloadImageFromNet(url,name):  
    #temp function- this functions download the images from the URLs      
    try :
        response = requests.get(url,timeout = 5)
        img = Image.open(BytesIO(response.content))
        img = img.convert('RGB')
        img.save(r'Images/'+name+r'.jpg')
    except Exception as e:
         print (e)   
    return 

def getImageClassFromJson ():
    #loading dataset from json
    with open(r'Dataset/face_detection.json','r') as f:
        x=[]        
        try: 
            x = json.load(f)
        except Exception as e:
            print (e)
    #print (x)
    return x

# def generate(train_tensors):
#     datagen = ImageDataGenerator(
#         width_shift_range=0.1,  
#         height_shift_range=0.1,  
#         horizontal_flip=True) 

#     datagen.fit(train_tensors)
#     return datagen
"""
if __name__ == '__main__':
    raw = input('Welcome to Tal network\r\n'+
    'For Training press t\r\n'+
    'For Accuracy test press a\r\n'+
    'Any other key to quit\r\n')
    char = raw.split()
    char[0] = char[0].upper()
    if char[0] == 'T':
        #training('saved_models/weights_model1.hdf5',50,20)
        training(None,100,20)
    elif char[0] == 'A':
        acc('saved_models/weights_model1.hdf5')
"""


"""
##visualization of data
#this function shows the images and rectangles from the dataset
if __name__ == '__main__':
    tragetHight = 224
    targetWidth = 224
    img_class = getImageClassFromJson()
 #   dModel = FaceDetectionModel().getModelInstance(BaseModel.VGG16)
    i = 0
    for data in img_class:        
        rectangles = []
        for anno in data['annotation']:
            #image original size
            width = anno['imageWidth']
            hight = anno['imageHeight']
            #calculating ratio
            rHight = tragetHight / hight
            rWidth = targetWidth / width
            #get 4 points of annotation
            x1 = anno['points'][0]['x']
            y1 = anno['points'][0]['y']
            x2 = anno['points'][1]['x']
            y2 = anno['points'][1]['y']
            #create rectangle
            rectangles.append(plt.Rectangle((width*rWidth*x1,hight*rHight*y1),(x2-x1)*width*rWidth,(y2-y1)*hight*rHight,fill = False, color = 'green'))
        img = Image.open(r"Dataset/Images/Train/"+str(i)+".jpg")
        img = img.resize((tragetHight,targetWidth))

        fig,axes = plt.subplots(1)
        plt.imshow(img)       
        #adding rectrangles
        for rec in rectangles:
           axes.add_patch(rec)
        plt.show(img)
        i=i+1

    #img_batch = np.expand_dims(img_to_array(img), axis=0)
    #dModel.preprocess_input(img_batch.copy())
    #dModel.predict()
    #download images
    # for i in range(len(x)):
    #     DownloadImageFromNet(x[i]['content'],str(i))
""" 

def DataPreprocess(tragetHight,targetWidth,trainSize, validationSize):
    #this functions process the data into a unified structure and resolution
    img_class = getImageClassFromJson()    
    i = 0
    rectangles = [[]]
    img_batches ={}
    appendFlag = False
    for data in img_class:
        if (appendFlag):
            rectangles.append([]) 
        appendFlag = True           
        for anno in data['annotation']:
            #image original size
            width = anno['imageWidth']
            hight = anno['imageHeight']
            #calculating ratio
            rHight = tragetHight / hight
            rWidth = targetWidth / width
            #get 4 points of annotation with accordance to image resize to (224,224)
            x1 = anno['points'][0]['x']*width*rWidth
            y1 = anno['points'][0]['y']*hight*rHight
            x2 = anno['points'][1]['x']*width*rWidth
            y2 = anno['points'][1]['y']*hight*rHight
            #save rectangles points                
            rectangles[i].append(anno['points'])        
        if (i<trainSize):
            img = Image.open(r"Dataset/Images/Train/"+str(i)+".jpg")
            pName='train'
        elif(i<trainSize+validationSize):
            img = Image.open(r"Dataset/Images/Validation/"+str(i)+".jpg")
            pName='validation'
        else:
            img = Image.open(r"Dataset/Images/Test/"+str(i)+".jpg")
            pName='test'
        img = img.resize((tragetHight,targetWidth))
        img_batches[pName] = np.expand_dims(img_to_array(img), axis=0)
        i=i+1
    return img_batches,rectangles    

if __name__ == '__main__':
    tragetHight = 224
    targetWidth = 224
    trainSize = 285
    validationSize = 76
    img_batches,rectangles = DataPreprocess(tragetHight = tragetHight,targetWidth = targetWidth,trainSize = trainSize, validationSize = validationSize)
    params = {'dim': (tragetHight,targetWidth,3),
          'batch_size': 64,
          'n_channels': 1,
          }
    dModel = FaceDetectionModel().getBaseModelInstance(BaseModel.VGG16)  
    training_generator = DataGenerator(img_batches['train'], rectangles, **params)

    dModel.fit_generator(generator=training_generator)
                        #     dModel.fit_generator(generator=training_generator,
                    # use_multiprocessing=True,
                    # workers=6,steps_per_epoch = 1)
    #dModel.preprocess_input(img_batch.copy())
    #dModel.predict()
    #download images
    # for i in range(len(x)):
    #     DownloadImageFromNet(x[i]['content'],str(i))