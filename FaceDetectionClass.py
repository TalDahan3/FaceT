import keras
from keras.applications import resnet50, vgg16
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
        return super().__init__(*args, **kwargs)       

    def getBaseModelInstance(self,modelChoose):
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
    def getRpnModel(self):
        return None ###dummy

    def _getRpnModel (self):
        


# class FaceDetectionModel:
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         resnet50Model = vgg16.VGG16(weights = None,include_top = False, pooling = 'max')
#         resnet50Model.summary()
#         return resnet50Model



