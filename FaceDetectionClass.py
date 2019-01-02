from enum import Enum

import keras
from keras.applications import resnet50, vgg16
from keras.layers import (Activation, Conv2D, Dense, Dropout, Flatten,
                          GlobalAveragePooling2D, MaxPooling2D, TimeDistributed)
from keras.models import Model,Sequential

from RoiPoolingLayer import RoiLayer


class BaseModel(Enum):
    NA=0
    START=1
    VGG16=1
    Resnet50=2
    END=2


### Faster -RCNN model
class FaceDetectionModel:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)       

    def getBaseModelInstance(self,modelChoose,weights_name):
        if (modelChoose == BaseModel.NA):
            return None,"Invalid model chose"
        if (weights_name == ""):
            weights_name = None
        if modelChoose == BaseModel.Resnet50:
            resnet50Model = resnet50.ResNet50(weights = weights_name,include_top = False, pooling = 'max',input_shape=(224, 224, 3))
            resnet50Model.summary()
            return resnet50Model
        else:
            vgg16Model = vgg16.VGG16(weights = None,include_top = False, pooling = None,input_shape=(224, 224, 3))
            vgg16Model.summary()
            model = Sequential()
            for layer in vgg16Model.layers[:-1]:
                model.add(layer)

            model.summary()
            return model

    def getRpnModel (self,baseModel,num_of_anchors):
        """
        Algorithm : https://medium.com/@tanaykarmarkar/region-proposal-network-rpn-backbone-of-faster-r-cnn-4a744a38d7f9
        """

        
        base = Conv2D(filters = 224,kernel_size = 3, strides = 1, activation = 'relu',name = 'rpn_conv1')
        x= keras.layers.concatenate([base,baseModel])
        x.summary()
        classes = Conv2D(num_of_anchors, kernel_size = 1, activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(base)
        regr = Conv2D(num_of_anchors * 4, kernel_size = 1, activation='linear', kernel_initializer='zero', name='rpn_out_regress')(base)
        return base, classes, regr
 
    def getDetectorClassifier(self, baseModel, input_rois, num_of_rois, output_classes, pooling_size = 7 ):
        out_roi_pool = RoiLayer(num_of_rois,pooling_size)
        out = TimeDistributed(Flatten(name='flatten'))(out_roi_pool)
        out = TimeDistributed(Dense(4096, activation='relu', name='detector_fc1'))(out)
        #needed??? TODO
        out = TimeDistributed(Dropout(0.5))(out)
        out = TimeDistributed(Dense(4096, activation='relu', name='detector_fc2'))(out)
        #needed??? TODO
        out = TimeDistributed(Dropout(0.5))(out)

        out_class = TimeDistributed(Dense(output_classes, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(output_classes))(out)
        # note: no regression target for bg class
        out_regr = TimeDistributed(Dense(4 * (output_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(output_classes))(out)

        return [out_class, out_regr]

    # def getFullModel(self, BaseModel, anchors,output_classes = 2, pooling_size = 7):


