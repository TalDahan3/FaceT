import keras
from keras.applications import resnet50, vgg16
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense, Activation
from enum import Enum

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

    def _getBaseModelInstance(self,modelChoose):
        if (modelChoose == BaseModel.NA):
            return None,"Invalid model chose"
        if modelChoose == BaseModel.Resnet50:
            resnet50Model = resnet50.ResNet50(weights = None,include_top = False, pooling = 'max',input_shape=(224, 224, 3))
            resnet50Model.summary()
            return resnet50Model
        else:
            vgg16Model = vgg16.VGG16(weights = None,include_top = False, pooling = 'max',input_shape=(224, 224, 3))
            vgg16Model.summary()
            return vgg16Model

    def _getRpnModel (self,baseModel,num_of_anchors):
        """
        Algorithm : https://medium.com/@tanaykarmarkar/region-proposal-network-rpn-backbone-of-faster-r-cnn-4a744a38d7f9
        """
        base = Conv2D(filters = 224,kernel_size = 3, strides = 1, activation = 'relu',name = 'rpn_conv1')(baseModel)
        classes = Conv2D(num_of_anchors, kernel_size = 1, activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(base)
        regr = Conv2D(num_of_anchors * 4, kernel_size = 1, activation='linear', kernel_initializer='zero', name='rpn_out_regress')(base)
        return base, classes, regr
 

        


# class FaceDetectionModel:
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         resnet50Model = vgg16.VGG16(weights = None,include_top = False, pooling = 'max')
#         resnet50Model.summary()
#         return resnet50Model



